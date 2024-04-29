import json
import os
import re
import random
import logging
import argparse
import torch
import torchmetrics
import evaluate
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import nlpaug.augmenter.word as naw

from rich.logging import RichHandler
from typing import Tuple
from textcls_tune import text_classifier,predict
from datasets import Dataset

from minigpt4.common.config import Config
from minigpt4.conversation.conversation import ChatInference
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


class CONST:
    answers=["screen","paper","away"]

def get_reject(answer):
    possible = [p for p in CONST.answers if p!=answer]
    return possible

def setup_seeds(seed:int)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def init_logger() -> None:
    logger = logging.getLogger("rich")
    logger.setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    FORMAT = "%(name)s[%(process)d] " + \
        "%(processName)s(%(threadName)s) " + \
        "%(module)s:%(lineno)d  %(message)s"

    formatter = logging.Formatter(
        FORMAT,
        datefmt="%Y%m%d %H:%M:%S"
    )
    logging.basicConfig(
        level="NOTSET", format=FORMAT, handlers=[RichHandler()]
    )

    ch = logging.StreamHandler()
    ch.setLevel(print)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    print("Initializing ok.")


def load_chat(
    args:argparse.ArgumentParser,
    annotate:bool=False,
)->Tuple[ChatInference,torch.nn.Module]:
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = int(args.gpu_id.split(':')[-1])
    model_config.classes = args.classes
    model_config.labels = args.labels
    print("LLAMA MODEL PATH",args.llama_model)
    if args.llama_model is not None:
        model_config.llama_model = args.llama_model

    print(f"llama model: {model_config.llama_model}")

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device='{}'.format(args.gpu_id))
    
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = ChatInference(model, vis_processor, device='{}'.format(args.gpu_id),annotate=annotate,conv_rec=3)
    
    chat.model = chat.model.to(device='{}'.format(args.gpu_id))
    print(f"USING DEVICE - {chat.model.device}")
    # https://stackoverflow.com/questions/69546459/convert-hydra-omegaconf-config-to-python-nested-dict-list
    return chat,model

def evaluate_caption(
    captions:list, 
    responses:list,
    metric:evaluate.load,
    scores:torch.Tensor,
    thres:int=0.5
)->list:
    for i,cap in enumerate(captions):
        # Calculate METEOR score for each model response compared to captions
        score = metric.compute(
            predictions=[responses],
            references=[cap]
        )
        scores[i] = score['meteor']
    # Sort scores based on highest METEOR score
    pred = (scores > thres).int()
    return torch.argmax(pred) if scores.sum() > 0 else -1


def aug_res(
    res:str,
    n_aug:int=10,
)->list:
    # Initialize augmenter
    augmenter = naw.SynonymAug(aug_src='wordnet')
    # Augment data
    augs = [res]
    for _ in range(n_aug):
        aug = augmenter.augment(res)
        if not aug:
            aug = ' '
        augs.extend(aug)
    return augs

def keyword_eval(
    num_classes:int=3,
)->None:
    # init_logger()
    args = parse_args()
    setup_seeds(1000)
    chat,model = load_chat(args=args,annotate=False)    
    label = get_test_labels(
        label_path=args.label_path
    )
    torch.set_default_device(f'{args.gpu_id}')
    queries = 'Is the person looking straight at the screen? Is the person looking down at the paper? Is the person looking away?'
    f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=num_classes,average='micro').to(chat.model.device) # average=None for all classes eval
    pr_micro = torchmetrics.Precision(task="multiclass", average='micro', num_classes=num_classes).to(chat.model.device)
    re_micro = torchmetrics.Recall(task="multiclass", average='micro', num_classes=num_classes).to(chat.model.device)
    f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=num_classes,average='macro').to(chat.model.device)
    pr_macro = torchmetrics.Precision(task="multiclass", average='macro', num_classes=num_classes).to(chat.model.device)
    re_macro = torchmetrics.Recall(task="multiclass", average='macro', num_classes=num_classes).to(chat.model.device)
    
    inference_samples = len(os.listdir(args.test_dir))
    pred_table,target_table = torch.zeros(inference_samples).to(chat.model.device),torch.zeros(inference_samples).to(chat.model.device)

    for sample,img_f in enumerate(os.listdir(args.test_dir)): 
        if sample >= inference_samples:
            break
        
        image_id = img_f.split('.')[0]
        print(f"IMAGE - {image_id}")
        print(f"LABEL - {label[img_f.replace('.jpg','')]}")
        target_table[sample] = int(label[img_f.replace('.jpg','')][0]) 

        message = chat.upload_img(os.path.join(args.test_dir,img_f))
        print(f"MESSAGE - {message}")

        print(f"QUERY - {queries}")
        a = chat.answer(queries,repetition_penalty=1.0)[0] # 1.0 is no penalty
        print(f"ANSWER - {a}") 
         
        pred_table[sample] = target_table[sample]
        if label[img_f.replace('.jpg','')][1] not in a.lower():
            pred_table[sample] = (target_table[sample] - 1) % num_classes
        elif get_reject(label[img_f.replace('.jpg','')][1])[0] in a.lower() or get_reject(label[img_f.replace('.jpg','')][1])[1] in a.lower():
            pred_table[sample] = (target_table[sample] - 1) % num_classes

            # logging.debug(f"PROMPT - {chat.default_prompt}")
            # dpo_sample['rejected'] = a
        
        pred, target = pred_table[:sample+1],target_table[:sample+1]
        f1_a,pr_a,rec_a = f1_macro(pred,target), pr_macro(pred,target),re_macro(pred,target)
        f1_i,pr_i,rec_i = f1_micro(pred,target), pr_micro(pred,target),re_micro(pred,target)
        print("FREE FORM MACRO: f1 - {:.3f}, pr - {:.3f}, re - {:.3f}".format(f1_a.item(),pr_a.item(),rec_a.item()))
        print("FREE FORM MICRO: f1 - {:.3f}, pr - {:.3f}, re - {:.3f}".format(f1_i.item(),pr_i.item(),rec_i.item()))
        

    

def bert_eval(
    num_classes:int=3,
    desc_data:list=[],
    pope_data:list=[],
)->None:
    # init_logger()
    args = parse_args()
    setup_seeds(1000)
    chat,model = load_chat(args=args,annotate=False) 
    label = get_test_labels(
        label_path=args.label_path
    )
    tokenizer, classifier = text_classifier(model_name="/home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/distillbert")

    # captions = [
    #     'Person is looking straight at the screen',
    #     'Person is looking down at the paper',
    #     'Person is looking away'
    # ]
    desc_prompt = "Describe this image in detail."
    queries = 'Is the person looking straight at the screen? Is the person looking down at the paper? Is the person looking away?'
    f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=num_classes,average='micro').to(chat.model.device) # average=None for all classes eval
    pr_micro = torchmetrics.Precision(task="multiclass", average='micro', num_classes=num_classes).to(chat.model.device)
    re_micro = torchmetrics.Recall(task="multiclass", average='micro', num_classes=num_classes).to(chat.model.device)
    
    f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=num_classes,average='macro').to(chat.model.device)
    pr_macro = torchmetrics.Precision(task="multiclass", average='macro', num_classes=num_classes).to(chat.model.device)
    re_macro = torchmetrics.Recall(task="multiclass", average='macro', num_classes=num_classes).to(chat.model.device)
    
    # meteor = evaluate.load('meteor')
    # scores = torch.zeros(len(captions))
    
    inference_samples = len(os.listdir(args.test_dir))
    pred_table,target_table = torch.zeros(inference_samples).to(chat.model.device),torch.zeros(inference_samples).to(chat.model.device)
    # meteor_scores = torch.zeros(inference_samples)
    eval_df = pd.DataFrame(
        np.zeros((inference_samples,5),dtype=object),
    ) 
    eval_df.iloc[:,1] = queries
    for sample,img_f in enumerate(os.listdir(args.test_dir)): 
        if sample >= inference_samples:
            break
        
        image_id = img_f.split('.')[0]
        print(f"IMAGE - {image_id}")
        print(f"LABEL - {label[image_id]}")
        target_table[sample] = int(label[image_id][0]) 

        message = chat.upload_img(os.path.join(args.test_dir,img_f))
        print(f"MESSAGE - {message}")

        print(f"PROMPT - {chat.default_prompt}")
        a = chat.answer(queries,repetition_penalty=1.5,temperature=0.1)[0] # 1.0 is no penalty
        print(f"ANSWER - {a}") 
        
        # pred = evaluate_caption(captions,a,meteor,scores)
        pred = predict(a,tokenizer,classifier)
        pred_table[sample] = pred
        if pred_table[sample] !=  target_table[sample]:
            pred_table[sample] = (target_table[sample] - 1) % num_classes
            detail_msg = chat.answer(desc_prompt,repetition_penalty=1.5,temperature=0.1)[0]
            desc_data.append({
                'image_id':image_id, 
                'chosen': aug_res(res=label[image_id][-1],n_aug=5),
                'rejected':aug_res(res=detail_msg,n_aug=5),
            })
            pope_data.append({
                'image_id':image_id, 
                'chosen': label[image_id][-1],
                'reject':a,
                'answer':label[image_id][1],
                'question':queries
            })
            
        # meteor_scores[sample] = scores[pred]
        # print("METEOR SCORE - {:.3f}".format(meteor_scores[:sample+1].sum()/(sample+1)))
        
        pred, target = pred_table[:sample+1],target_table[:sample+1]
        f1_a,pr_a,rec_a = f1_macro(pred,target), pr_macro(pred,target),re_macro(pred,target)
        f1_i,pr_i,rec_i = f1_micro(pred,target), pr_micro(pred,target),re_micro(pred,target)
        print("FREE FORM MACRO: f1 - {:.3f}, pr - {:.3f}, re - {:.3f}".format(f1_a.item(),pr_a.item(),rec_a.item()))
        print("FREE FORM MICRO: f1 - {:.3f}, pr - {:.3f}, re - {:.3f}".format(f1_i.item(),pr_i.item(),rec_i.item()))
        
        
        eval_df.iloc[sample,0] = image_id
        eval_df.iloc[sample,2] = a
        eval_df.iloc[sample,3] = pred_table[sample].item()
        eval_df.iloc[sample,4] = target_table[sample].item()
    
    if args.dpo_pairs:
        with open(args.desc_data,'w') as f:
            json.dump(desc_data,f,indent=4)
        
        with open(args.pope_data,'w') as f:
            json.dump(pope_data,f,indent=4)
    
    out_file = args.test_dir.split('/')[-2]
    eval_df.columns = ['image_id','question','answer','pred','label']
    eval_df.to_csv(f'/home/tony/HA-DPO/csv/{out_file}.csv',index=False)


def get_test_labels(
    label_path:str
)->dict:
    label = {}
    classes = np.array([
        [0,'screen',"Person is looking straight at the screen"],
        [1,'paper',"Person is looking down at the paper"],
        [2,'away',"Person is looking away"]
    ])
    with open(label_path,'r') as f:
        captions = json.load(f)
        for pair in captions['annotations']:
            label[pair['image_id']] = classes[[
                ('screen' in pair['caption']),
                ('paper' in pair['caption']),
                ('away' in pair['caption'])
            ]][0].tolist()
    save = open(os.path.join('/'.join(label_path.split('/')[:-1]),'filter_cap_mod.json'),'w')
    json.dump(label,save,indent=4)
    save.close()
    return label
    
def parse_args():
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument('-cfg-path',"--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument('-classes',"--classes",type=int, default=3,help="The number of classes")
    parser.add_argument("--gpu-id", type=str, default='cuda:0', help="specify the gpu to load the model.")
    parser.add_argument(
        "--test-dir", 
        type=str, 
        default='/data/tony/iccvw/testing/image', 
        help="directory of images to evaluate"
    )
    parser.add_argument(
        "--label-path", 
        type=str, 
        default='/data/tony/iccvw/testing/filter_cap.json', 
        help="path to filter_cap.json"
    )
    parser.add_argument(
        "--desc-data", 
        type=str, 
        default='/home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/sed_desc_data.json', 
        help="outfile location for desc data"
    )
    parser.add_argument(
        "--pope-data", 
        type=str, 
        default='/home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/sed_pope_data.json', 
        help="outfile location for pope data"
    )
    parser.add_argument(
        '-labels','--labels', 
        nargs='+',
        default=[
            "Person is looking at the screen.",
            "Person is looking at the paper.",
            "Person is looking away."
        ] ,
        help='Input captions of classes, order matters. Keep same order of captions for inference.', 
        required=False
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--llama-model", 
        default=None, 
        help="path to language model file."
    )
    parser.add_argument(
        "--dpo-pairs", 
        action='store_true',
        help="whether to save desc and pope data pairs"
    )
    args = parser.parse_args()
    return args


def main()->None:
    # keyword_eval()
    bert_eval()

if __name__ == "__main__":
    """
    lu runs batch size of 6
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:0 \
        --test-dir /home/tony/HA-DPO/ha_dpo/data/lubal_sed_training/image \
        --label-path /home/tony/HA-DPO/ha_dpo/data/lubal_sed_training/filter_cap.json\
        --desc-data /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/sed_desc_data.json \
        --pope-data /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/sed_pope_data.json \
        --dpo-pairs > /home/tony/HA-DPO/logs/minigpt4_train_bal_sed.txt
        
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:1 \
        --test-dir /home/tony/HA-DPO/ha_dpo/data/lubal_sed_testing/image \
        --label-path /home/tony/HA-DPO/ha_dpo/data/lubal_sed_testing/filter_cap.json \
        --llama-model /home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/merged_sed_minigpt4_hakto > /home/tony/HA-DPO/logs/minigpt4_eval_sed_hakto.txt
        
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:0 \
        --test-dir /home/tony/luraw_sed_testing/image \
        --label-path /home/tony/luraw_sed_testing/filter_cap_raw.json \
        --llama-model /home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/merged_sed_minigpt4_hakto > /home/tony/HA-DPO/logs/minigpt4_eval_raw_hakto.txt
      
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:1 \
        --test-dir /home/tony/daisee/image \
        --label-path /home/tony/daisee/filter_cap.json \
        --llama-model /home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/merged_sed_minigpt4_hakto > /home/tony/HA-DPO/logs/minigpt4_eval_daisee_hakto.txt
        
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:0 \
        --test-dir /home/tony/handpicked/image \
        --label-path /home/tony/handpicked/filter_cap.json \
        --llama-model /home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/merged_sed_minigpt4_hakto > /home/tony/HA-DPO/logs/minigpt4_eval_handpicked_hakto.txt
    
    
      
       
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:0 \
        --test-dir /home/tony/HA-DPO/ha_dpo/data/lubal_sed_testing/image \
        --label-path /home/tony/HA-DPO/ha_dpo/data/lubal_sed_testing/filter_cap.json \
        --llama-model /home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/merged_sed_minigpt4_hadpo > /home/tony/HA-DPO/logs/minigpt4_eval_sed_hadpo.txt 
        
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:0 \
        --test-dir /home/tony/luraw_sed_testing/image \
        --label-path /home/tony/luraw_sed_testing/filter_cap_raw.json \
        --llama-model /home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/merged_sed_minigpt4_hadpo > /home/tony/HA-DPO/logs/minigpt4_eval_raw_hadpo.txt 
        
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:1 \
        --test-dir /home/tony/daisee/image \
        --label-path /home/tony/daisee/filter_cap.json \
        --llama-model /home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/merged_sed_minigpt4_hadpo > /home/tony/HA-DPO/logs/minigpt4_eval_daisee_hadpo.txt
        
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:0 \
        --test-dir /home/tony/handpicked/image \
        --label-path /home/tony/handpicked/filter_cap.json \
        --llama-model /home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/merged_sed_minigpt4_hadpo > /home/tony/HA-DPO/logs/minigpt4_eval_handpicked_hadpo.txt        
        
        
        
        
        
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:0 \
        --test-dir /home/tony/HA-DPO/ha_dpo/data/lubal_sed_testing/image \
        --label-path /home/tony/HA-DPO/ha_dpo/data/lubal_sed_testing/filter_cap.json > /home/tony/HA-DPO/logs/minigpt4_eval_sed_base.txt
    
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:1 \
        --test-dir /home/tony/daisee/image \
        --label-path /home/tony/daisee/filter_cap.json > /home/tony/HA-DPO/logs/minigpt4_eval_daisee_base.txt
    
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:1 \
        --test-dir /home/tony/handpicked/image \
        --label-path /home/tony/handpicked/filter_cap.json > /home/tony/HA-DPO/logs/minigpt4_eval_handpicked_base.txt
        
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:1 \
        --test-dir /home/tony/luraw_sed_testing/image \
        --label-path /home/tony/luraw_sed_testing/filter_cap_raw.json > /home/tony/HA-DPO/logs/minigpt4_eval_raw_base.txt
        
        
        
        
        
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:0 \
        --test-dir /home/tony/HA-DPO/ha_dpo/data/lubal_sed_testing/image \
        --label-path /home/tony/HA-DPO/ha_dpo/data/lubal_sed_testing/filter_cap.json > /home/tony/HA-DPO/logs/minigpt4_eval_sed_general.txt
    
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:1 \
        --test-dir /home/tony/daisee/image \
        --label-path /home/tony/daisee/filter_cap.json > /home/tony/HA-DPO/logs/minigpt4_eval_daisee_general.txt
    
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:1 \
        --test-dir /home/tony/handpicked/image \
        --label-path /home/tony/handpicked/filter_cap.json > /home/tony/HA-DPO/logs/minigpt4_eval_handpicked_general.txt

    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:1 \
        --test-dir /home/tony/luraw_sed_testing/image \
        --label-path /home/tony/luraw_sed_testing/filter_cap_raw.json > /home/tony/HA-DPO/logs/minigpt4_eval_raw_general.txt   
        
        
        
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:0 \
        --test-dir /home/tony/HA-DPO/ha_dpo/data/lubal_sed_testing/image \
        --label-path /home/tony/HA-DPO/ha_dpo/data/lubal_sed_testing/filter_cap.json > /home/tony/HA-DPO/logs/minigpt4_eval_sed_bal.txt
    
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:1 \
        --test-dir /home/tony/daisee/image \
        --label-path /home/tony/daisee/filter_cap.json > /home/tony/HA-DPO/logs/minigpt4_eval_daisee_sed.txt
    
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:1 \
        --test-dir /home/tony/handpicked/image \
        --label-path /home/tony/handpicked/filter_cap.json > /home/tony/HA-DPO/logs/minigpt4_eval_handpicked_sed.txt
    
    python test_student.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cuda:1 \
        --test-dir /home/tony/luraw_sed_testing/image \
        --label-path /home/tony/luraw_sed_testing/filter_cap_raw.json > /home/tony/HA-DPO/logs/minigpt4_eval_raw_sed.txt
    
    """
    main() 

    