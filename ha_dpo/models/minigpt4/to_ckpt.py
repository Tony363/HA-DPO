import torch
import os
import copy
from safetensors.torch import load_file
from test_student import load_chat,parse_args

def main()->None:
    args = parse_args()
    chat,model = load_chat(args=args,annotate=False) 
    print("ARCH")
    for name,param in model.named_parameters():
        print(name)
    directory = '/home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/merged_sed_minigpt4_hadpo'
    # directory = '/home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/sed_minigpt4_hadpo'
    weights = {
        'model':{}
    }
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has the .safetensors extension
        if filename.endswith('.safetensors'):
            # The full path to the file
            file_path = os.path.join(directory, filename)
            cw = load_file(file_path)    
            weights['model'].update(cw)
            
    print("DPO WEIGHTS")
    header = ('llama_proj','llama_model',)
    layers = copy.deepcopy(list(weights['model'].keys()))
    for name in layers:
        layer = name.split('.')
        weights['model'][f'{header[int(len(layer) != 2)]}.{name}'] = weights['model'].pop(name)
    torch.save(weights, os.path.join(directory,'hadpo_minigpt4_sed.pth'))
    
    
    

if __name__ == "__main__":
    """
    python to_ckpt.py \
        --cfg-path /home/tony/HA-DPO/ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml  \
        --gpu-id cpu > /home/tony/HA-DPO/logs/ha_dpo_weights.txt
    """
    main()