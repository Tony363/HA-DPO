import torch
import os
from safetensors.torch import load_file

def main()->None:
    # directory = '/home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/merged_sed_minigpt4_hadpo'
    directory = '/home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/sed_minigpt4_hadpo'
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
            # print(cw.keys(),type(cw))
            weights['model'].update(cw)
            
    torch.save(weights, os.path.join(directory,'hadpo_minigpt4_sed.pth'))
    

if __name__ == "__main__":
    main()