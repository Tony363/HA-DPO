import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments,Trainer
from transformers.trainer_callback import EarlyStoppingCallback

import nlpaug.augmenter.word as naw
from datasets import Dataset

def setup_seeds(seed:int)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def augment_sed_data(
    n_aug:int=10,
    sed_out_dst:str='/home/tony/HA-DPO/ha_dpo/data/distillbert_sed_data.csv'
)->Dataset:
    # Original data
    data = {
        'text': [
            "Person is looking straight at the screen",
            "Person is looking down at the paper",
            "Person is looking away"
        ],
        'label': [0, 1, 2]  # 0: screen, 1: paper, 2: away
    }

    # Initialize augmenter
    augmenter = naw.SynonymAug(aug_src='wordnet')

    # Augment data
    augmented_texts, augmented_labels = [], []
    for text, label in zip(data['text'], data['label']):
        for _ in range(n_aug):  # Generate 5 augmented versions per sample
            augmented_texts.extend(augmenter.augment(text))
            augmented_labels.append(label)

    # Combine original and augmented data
    data['text'].extend(augmented_texts)
    data['label'].extend(augmented_labels)
    # Convert to Hugging Face dataset
    dataset = Dataset.from_dict(data)
    dataset.to_csv(sed_out_dst, index=False)
    return dataset

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

def text_classifier(
    model_name:str="distilbert-base-uncased",
    num_labels:int=3,
)->tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    return tokenizer,model


def predict(
    res:str,
    tokenizer:DistilBertTokenizer,
    model:DistilBertForSequenceClassification
)->int:
    inputs = tokenizer(res,return_tensors='pt',padding="max_length", truncation=True).to(model.device)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1)
    return prediction.item()

def main(
    out_dst:str='minigpt4/output/distillbert'    
)->None:
    setup_seeds(42)
    ds = augment_sed_data(n_aug=100).map(tokenize_function,batched=True)
    training_args = TrainingArguments(
        output_dir=out_dst,
        evaluation_strategy="epoch",
        save_strategy="epoch",        # Save the model at the end of each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=16*2,
        per_device_eval_batch_size=16*2,
        num_train_epochs=100,
        weight_decay=0.01,
        load_best_model_at_end=True,  # This will load the best model at the end of training
        # metric_for_best_model='acc',
    )
    train_ds, test_ds = ds.train_test_split(test_size=0.2).values()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.01)]
    )
    trainer.train()
    trainer.evaluate()
    model.save_pretrained(f'{out_dst}')
    tokenizer.save_pretrained(f'{out_dst}')
    
    evaluate()
    return

def evaluate()->None:
    captions = [
        'Person is looking straight at the screen',
        'Person is looking down at the paper',
        'Person is looking away'
    ]
    tokenizer,model = text_classifier(model_name="/home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/distillbert")    
    for res in captions:
        prediction = predict(res,tokenizer,model)
        print(prediction)

if __name__ == "__main__":
    tokenizer,model = text_classifier()
    main()
