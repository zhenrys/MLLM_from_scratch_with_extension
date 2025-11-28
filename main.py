# main.py

import argparse
from utils.config_parser import parse_config

def main():
    parser = argparse.ArgumentParser(description="Multimodal Transformer from Scratch")
    parser.add_argument("--task", type=str, required=True, 
                        choices=['train_vit', 'predict_vit', 'train_llm', 'generate_text', 'train_mllm', 'inference_mllm', "train_rl_mllm"],
                        help="Task to run (e.g., train_vit, predict_vit, train_llm)")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")

    args = parser.parse_args()

    # 解析配置文件
    try:
        config = parse_config(args.config)
    except Exception as e:
        print(f"Error parsing config file: {e}")
        return

    print(f"Running task: {args.task}")
    print(f"Using config: {args.config}")

    # --- 任务分发器 ---
    if args.task == "train_vit":
        from vision_transformer.train_vit import train
        train(config)
    
    elif args.task == "predict_vit":
        from vision_transformer.predict_vit import predict
        predict(config)
        
    elif args.task == "train_llm":
        from language_model.train_llm import train
        train(config)
        
    elif args.task == "generate_text":
        from language_model.generate_text import generate
        generate(config)
        
    elif args.task == "train_mllm":
        from multimodal_model.train_mllm import train
        train(config)
        
    elif args.task == "inference_mllm":
        from multimodal_model.inference_mllm import inference
        inference(config)
        
    elif args.task == "train_rl_mllm":
        from multimodal_model.train_rl_mllm import rl_finetune
        rl_finetune(config)
        
    else:
        print(f"Task '{args.task}' is not recognized.")


if __name__ == "__main__":
    main()