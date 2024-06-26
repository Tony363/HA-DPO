import os
os.environ["WANDB_PROJECT"]="ha-dpo"

import yaml
import json
import copy
import torch
import random
import argparse
import numpy as np
from PIL import Image
from argparse import Namespace
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from torch.utils.data import ConcatDataset

from peft.peft_model import PeftModelForCausalLM
from transformers import TrainerCallback
from transformers import HfArgumentParser, TrainingArguments

import minigpt4.tasks as tasks
from minigpt4.common.config import Config

from ha_dpo.trainer.minigpt4_dpo_trainer import MiniGPT4DPOTrainer
from dpo_dataset import PopeDataset, AugmentedCaptionDataset, CCSBUAlignDataset

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    
    # model config path
    cfg_path: str = field(metadata={"help": "path to configuration file."})
    
    # data parameters
    ccsbualign_data_path: Optional[str] = field(default=None, metadata={"help": "path of the CCSBUAlign data."})
    desc_train_data_path: Optional[str] = field(default=None, metadata={"help": "path of the description positive-negative data."})
    pope_train_data_path: Optional[str] = field(default=None, metadata={"help": "path of the pope-format positive-negative data."})
    vg_path: Optional[str] = field(default="", metadata={"help": "path of visual genome annotation file."})
    
    # hyper-parameters
    seed: Optional[int] = field(default=42, metadata={"help": "training and data seed."})
    gamma: Optional[float] = field(default=1.0, metadata={"help": "weight factor of auxilary language modeling task."})
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    auxilary: Optional[bool] = field(default=False, metadata={"help": "whether to use auxilary task during DPO."})
    reference_free: Optional[str] = field(default='sigmoid', metadata={"help": "If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses."})
    loss_type: Optional[str] = field(default='sigmoid', metadata={"help": "the loss type for DPO."})
    label_smoothing: Optional[float] = field(default=0, metadata={"help": "[cDPO](https://ericmitchell.ai/cdpo.pdf) report that should be between 0 and 0.5"})


    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "maximum value of gradient norm"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=True, metadata={"help": "whether to find unused parameters. set to False when `gradient_checkpointing` is False."}
    )
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[float] = field(default=-1, metadata={"help": "number of trained eppchs."})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=-1, metadata={"help": "the saving frequency"})
    evaluation_strategy: Optional[str] = field(default='no', metadata={"help": "the evaluation strategy"})
    eval_steps: Optional[float] = field(default=None, metadata={"help": "the evaluation frequency"})
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    run_name: Optional[str] = field(default="minigpt4", metadata={"help": "name of the run"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    
    # lora parameters
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=2, metadata={"help": "the lora r parameter"})
    lora_target_modules: Optional[list[str]] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"], metadata={"help": "the lora modules"})
    freeze_llama_proj: Optional[bool] = field(default=True, metadata={"help": "whether to freeze llama_proj module"})
    
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


# callback used to save model
# !HACK: wandb upload failed!
class MyCallback(TrainerCallback):
    "A callback that prints a message at the end of training"
    def on_train_end(self, args, state, control, **kwargs):
        # save model
        if "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0:
            # save lora weights
            if isinstance(kwargs['model'].llama_model, PeftModelForCausalLM):
                kwargs['model'].llama_model.save_pretrained(args.output_dir)
            # if llama_proj is not frozen, save llama_proj
            if not kwargs['model'].freeze_llama_proj:
                state_dict = kwargs['model'].state_dict()
                for k in list(state_dict.keys()):
                    if "llama_proj" not in k:
                        del state_dict[k]
                torch.save({"model":state_dict}, os.path.join(args.output_dir, "llama_proj.bin"))
        print("ARGS!!!!",args)
        # save training arguments
        if "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0:
            print("Save model in the end of training")
            with open(os.path.join(args.output_dir, "training_args.yaml"), "w") as f:
                yaml.dump(args, f)
                
                
  
    
    
def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    cfg_dict = {'cfg_path': script_args.cfg_path, 'options': None}
    cfg = Config(Namespace(**cfg_dict))
    
    # set dpo model parameters
    cfg.config.model.lora_r = script_args.lora_r
    cfg.config.model.lora_alpha = script_args.lora_alpha
    cfg.config.model.lora_dropout = script_args.lora_dropout
    cfg.config.model.lora_target_modules = script_args.lora_target_modules
    cfg.config.model.freeze_llama_proj = script_args.freeze_llama_proj
    
    ref_cfg = copy.deepcopy(cfg)
    ref_cfg.config.model.lora_r = 0 # no lora in reference model
    ref_cfg.config.model.freeze_llama_proj = True # no lora in reference model
    
    # model & dataset
    # policy
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    # reference
    task_ref = tasks.setup_task(ref_cfg)
    ref_model = task_ref.build_model(ref_cfg)
    
    tokenizer = model.llama_tokenizer
    
    if script_args.auxilary:
        auxilary_dataset = CCSBUAlignDataset(
            ccsbualign_data_path = script_args.ccsbualign_data_path,
            cfg = cfg.config,
        )
    else:
        auxilary_dataset = None
    desc_train_dataset = AugmentedCaptionDataset(
        data_path = script_args.desc_train_data_path,
        vg_path = script_args.vg_path,
        cfg = cfg.config,
        seed = script_args.seed,
        auxilary_dataset = auxilary_dataset,
    )
    pope_train_dataset = PopeDataset(
        data_path = script_args.pope_train_data_path,
        vg_path = script_args.vg_path,
        cfg = cfg.config,
        auxilary_dataset = auxilary_dataset,
    )
    train_dataset = ConcatDataset([desc_train_dataset, pope_train_dataset])
    
    # if not use gradient_checkpointing, do not set ddp_find_unused_parameters
    if not script_args.gradient_checkpointing:
        script_args.ddp_find_unused_parameters = False
    
    # initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        ddp_find_unused_parameters=script_args.ddp_find_unused_parameters,
        learning_rate=script_args.learning_rate,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        max_grad_norm=script_args.max_grad_norm,
        seed=script_args.seed,
    )
    
    # initialize the DPO trainer
    dpo_trainer = MiniGPT4DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=script_args.beta,
        gamma=script_args.gamma,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        loss_type=script_args.loss_type,
    )
    
    # model save callback
    dpo_trainer.add_callback(MyCallback())
    
    dpo_trainer.train()
    
    # save script args
    with open(os.path.join(training_args.output_dir, "script_args.yaml"), "w") as f:
        yaml.dump(script_args, f)
    
if __name__ == "__main__":
    """
    WANDB_MODE=dryrun accelerate launch --main_process_port $RANDOM ha_dpo/models/minigpt4/train_dpo.py \
    --cfg_path ha_dpo/models/minigpt4/train_configs/minigpt4_llama2_stage3_dpo.yaml \
    --pope_train_data_path /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/sed_pope_data.json  \
    --desc_train_data_path /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/sed_desc_data.json  \
    --vg_path /home/tony/HA-DPO/ha_dpo/data/lubal_sed_training  \
    --auxilary True \
    --ccsbualign_data_path /home/tony/HA-DPO/ha_dpo/data/cc_sbu_align \
    --lora_r 64 \
    --gradient_checkpointing False \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --beta 0.1 \
    --gamma 0.5 \
    --reference_free False\
    --loss_type kto_pair \
    --label_smoothing 0 \
    --gradient_accumulation_steps 1 \
    --max_steps 1000 \
    --output_dir ha_dpo/models/minigpt4/minigpt4/output/sed_minigpt4_hakto \
    --logging_steps 4 
    
    
    python ha_dpo/models/minigpt4/merge_peft_adapter.py \
    --adapter_model_name ha_dpo/models/minigpt4/minigpt4/output/sed_minigpt4_hakto\
    --base_model_name wangrongsheng/MiniGPT-4-LLaMA-7B\
    --output_name ha_dpo/models/minigpt4/minigpt4/output/merged_sed_minigpt4_hakto
    
    
    
    WANDB_MODE=dryrun accelerate launch --main_process_port $RANDOM ha_dpo/models/minigpt4/train_dpo.py \
    --cfg_path ha_dpo/models/minigpt4/train_configs/minigpt4_llama2_stage3_dpo.yaml \
    --auxilary True \
    --ccsbualign_data_path /home/tony/HA-DPO/ha_dpo/data/cc_sbu_align \
    --pope_train_data_path /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/pope_data.json \
    --desc_train_data_path /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/desc_data.json \
    --vg_path /home/tony/HA-DPO/ha_dpo/data/VG \
    --lora_r 64 \
    --gradient_checkpointing False \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --beta 0.1 \
    --gamma 0.5 \
    --reference_free False\
    --loss_type kto_pair \
    --label_smoothing 0 \
    --gradient_accumulation_steps 4 \
    --max_steps 1000 \
    --output_dir 'ha_dpo/models/minigpt4/minigpt4/output/minigpt4_kto' \
    --logging_steps 4    
      
    python ha_dpo/models/minigpt4/merge_peft_adapter.py \
    --adapter_model_name ha_dpo/models/minigpt4/minigpt4/output/minigpt4_kto \
    --base_model_name wangrongsheng/MiniGPT-4-LLaMA-7B \
    --output_name ha_dpo/models/minigpt4/minigpt4/output/merged_minigpt4_hakto
      
      
    WANDB_MODE=dryrun accelerate launch --main_process_port $RANDOM ha_dpo/models/minigpt4/train_dpo.py \
    --cfg_path ha_dpo/models/minigpt4/train_configs/minigpt4_llama2_stage3_dpo.yaml \
    --auxilary True \
    --ccsbualign_data_path /home/tony/HA-DPO/ha_dpo/data/cc_sbu_align \
    --pope_train_data_path /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/pope_data.json \
    --desc_train_data_path /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/desc_data.json \
    --vg_path /home/tony/HA-DPO/ha_dpo/data/VG \
    --lora_r 64 \
    --gradient_checkpointing False \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --beta 0.1 \
    --gamma 0.5 \
    --reference_free False\
    --loss_type ipo \
    --label_smoothing 0 \
    --gradient_accumulation_steps 4 \
    --max_steps 1000 \
    --output_dir 'ha_dpo/models/minigpt4/minigpt4/output/minigpt4_ipo' \
    --logging_steps 4          
    
    python ha_dpo/models/minigpt4/merge_peft_adapter.py \
    --adapter_model_name ha_dpo/models/minigpt4/minigpt4/output/minigpt4_ipo \
    --base_model_name wangrongsheng/MiniGPT-4-LLaMA-7B \
    --output_name ha_dpo/models/minigpt4/minigpt4/output/merged_minigpt4_haipo
    
    
    WANDB_MODE=dryrun accelerate launch --main_process_port $RANDOM ha_dpo/models/minigpt4/train_dpo.py \
    --cfg_path ha_dpo/models/minigpt4/train_configs/minigpt4_llama2_stage3_dpo.yaml \
    --auxilary True \
    --ccsbualign_data_path /home/tony/HA-DPO/ha_dpo/data/cc_sbu_align \
    --pope_train_data_path /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/pope_data.json \
    --desc_train_data_path /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/desc_data.json \
    --vg_path /home/tony/HA-DPO/ha_dpo/data/VG \
    --lora_r 64 \
    --gradient_checkpointing False \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --beta 0.1 \
    --gamma 0.5 \
    --reference_free False\
    --loss_type hinge \
    --label_smoothing 0 \
    --gradient_accumulation_steps 4 \
    --max_steps 1000 \
    --output_dir 'ha_dpo/models/minigpt4/minigpt4/output/minigpt4_rso' \
    --logging_steps 4   
    
    
    python ha_dpo/models/minigpt4/merge_peft_adapter.py \
    --adapter_model_name ha_dpo/models/minigpt4/minigpt4/output/minigpt4_rso \
    --base_model_name wangrongsheng/MiniGPT-4-LLaMA-7B \
    --output_name ha_dpo/models/minigpt4/minigpt4/output/merged_minigpt4_harso
    
    WANDB_MODE=dryrun accelerate launch --main_process_port $RANDOM ha_dpo/models/minigpt4/train_dpo.py \
    --cfg_path ha_dpo/models/minigpt4/train_configs/minigpt4_llama2_stage3_dpo.yaml \
    --auxilary True \
    --ccsbualign_data_path /home/tony/HA-DPO/ha_dpo/data/cc_sbu_align \
    --pope_train_data_path /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/pope_data.json \
    --desc_train_data_path /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/desc_data.json \
    --vg_path /home/tony/HA-DPO/ha_dpo/data/VG \
    --lora_r 64 \
    --gradient_checkpointing False \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --beta 0.1 \
    --gamma 0.5 \
    --reference_free False\
    --loss_type sigmoid \
    --label_smoothing 0.5 \
    --gradient_accumulation_steps 4 \
    --max_steps 1000 \
    --output_dir 'ha_dpo/models/minigpt4/minigpt4/output/minigpt4_cdpo' \
    --logging_steps 4                            
    
    python ha_dpo/models/minigpt4/merge_peft_adapter.py \
    --adapter_model_name ha_dpo/models/minigpt4/minigpt4/output/minigpt4_cdpo \
    --base_model_name wangrongsheng/MiniGPT-4-LLaMA-7B \
    --output_name ha_dpo/models/minigpt4/minigpt4/output/merged_minigpt4_hacdpo  
    
    
    
    
    WANDB_MODE=dryrun accelerate launch --main_process_port $RANDOM ha_dpo/models/minigpt4/train_dpo.py \
    --cfg_path ha_dpo/models/minigpt4/train_configs/minigpt4_llama2_stage3_dpo.yaml \
    --auxilary True \
    --ccsbualign_data_path /home/tony/HA-DPO/ha_dpo/data/cc_sbu_align \
    --pope_train_data_path /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/pope_data.json \
    --desc_train_data_path /home/tony/HA-DPO/ha_dpo/data/hadpo/minigpt4/desc_data.json \
    --vg_path /home/tony/HA-DPO/ha_dpo/data/VG \
    --lora_r 64 \
    --gradient_checkpointing False \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --beta 0.1 \
    --gamma 0.5 \
    --reference_free False\
    --loss_type sigmoid \
    --label_smoothing 0 \
    --gradient_accumulation_steps 4 \
    --max_steps 1000 \
    --output_dir 'ha_dpo/models/minigpt4/minigpt4/output/minigpt4_dpo' \
    --logging_steps 4                            
    
    python ha_dpo/models/minigpt4/merge_peft_adapter.py \
    --adapter_model_name ha_dpo/models/minigpt4/minigpt4/output/minigpt4_dpo \
    --base_model_name wangrongsheng/MiniGPT-4-LLaMA-7B \
    --output_name ha_dpo/models/minigpt4/minigpt4/output/merged_minigpt4_hadpo                                                    
    """
    main()


                