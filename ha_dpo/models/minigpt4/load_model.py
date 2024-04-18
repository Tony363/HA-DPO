
import argparse
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.common.dist_utils import get_rank, init_distributed_mode

import logging
logging.basicConfig(level = logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="POPE Evaluation")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--coco-path", required=True, help="path to COCO2014 images.")
    parser.add_argument("--pope-path", required=True, help="path to POPE annotation file.")
    parser.add_argument("--set", required=True, help="which set of POPE, choose between random/popular/adv.")
    parser.add_argument("--llama-model", default=None, help="path to language model file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def main()->None:
    args = parse_args()
    cfg = Config(args)
    if args.llama_model is not None:
        cfg.model_cfg.llama_model = args.llama_model

    init_distributed_mode(cfg.run_cfg)

    model_config = cfg.model_cfg
    if args.llama_model is not None:
        model_config.llama_model = args.llama_model

    print(f"llama model: {cfg.model_cfg.llama_model}")

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)
    
if __name__ == "__main__":
    
    """
    torchrun --nproc-per-node 1 --master-port $RANDOM ha_dpo/models/minigpt4/load_model.py \
    --set popular \
    --cfg-path ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml \
    --llama-model /home/tony/HA-DPO/ha_dpo/models/minigpt4/minigpt4/output/merged_sed_minigpt4_hadpo \
    --pope-path ha_dpo/data/POPE \
    --coco-path /home/tony/HA-DPO/ha_dpo/data/coco2014 > logs/load_model_test.txt
    """
    main()