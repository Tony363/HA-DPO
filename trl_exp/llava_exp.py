import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset,load_dataset

from PIL import Image
import warnings

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
torch.set_default_device('cuda')  # or 'cpu'

def _data()->Dataset:
    # /home/tony/HA-DPO/ha_dpo/data/coco2014/val2014
    ds = load_dataset("ydshieh/coco_dataset_script", "2017", data_dir="./dummy_data/")
    return ds['train'],ds['validation'],ds['test']

def main()->None:
    # create model
    model = AutoModelForCausalLM.from_pretrained(
        'qnguyen3/nanoLLaVA',
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        'qnguyen3/nanoLLaVA',
        trust_remote_code=True)

    # text prompt
    prompt = 'Describe this image in detail'

    messages = [
        {"role": "user", "content": f'<image>\n{prompt}'}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

if __name__ == "__main__":
    # main()
    ds_train,ds_val,ds_test = _data()
    print(ds_val[0])