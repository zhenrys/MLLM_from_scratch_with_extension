# language_model/generate_text.py

import torch
import os
import torch.nn.functional as F

from language_model.tokenizer import CharacterTokenizer
from language_model.llm import GPTModel

@torch.no_grad()
def generate(config: dict):
    data_cfg = config['data_params']
    model_cfg = config['model_params']
    train_cfg = config['training_params']
    gen_cfg = config['generation_params']

    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else 'cpu')
    model_path = train_cfg['model_save_path']
    start_context = gen_cfg['start_context']
    max_new_tokens = gen_cfg['max_new_tokens']

    tokenizer = CharacterTokenizer()
    vocab_path = os.path.join(data_cfg['data_dir'], 'tinyshakespeare', 'vocab.json')
    try:
        tokenizer.load_vocab(vocab_path)
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {vocab_path}.")
        return
        
    model_cfg['vocab_size'] = tokenizer.get_vocab_size()
    
    model = GPTModel(vocab_size=model_cfg['vocab_size'], d_model=model_cfg['d_model'], num_layers=model_cfg['num_layers'], n_heads=model_cfg['n_heads'], d_ff=model_cfg['d_ff'], max_len=data_cfg['block_size'], dropout=model_cfg['dropout']).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_path}.")
        return

    model.eval()
    print(f"Model loaded from {model_path}")

    print("\n--- Generated Text ---")
    print(start_context, end='')

    context_tokens = tokenizer.encode(start_context)
    context = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # --- START OF STUDENT MODIFICATION (GENERATION LOOP) ---
    for _ in range(max_new_tokens):
        # 确保上下文长度不超过模型的 block_size
        context_cond = context[:, -data_cfg['block_size']:]
        
        # TODO: 1. 获取模型的 logits 输出
        logits = ...
        
        # TODO: 2. 提取最后一个时间步的 logits。
        #    - 我们只关心对下一个词元的预测，这对应于输入序列最后一个位置的输出。
        #    - `logits` 的形状是 (B, T, C)，我们需要的是形状为 (B, C) 的张量。
        #    - 提示: 使用切片 `[:, -1, :]`。
        logits_last_step = ...
        
        # TODO: 3. 将 logits 转换为概率分布。
        #    - 使用 `F.softmax` 函数在最后一个维度 (词汇表维度) 上操作。
        probs = ...
        
        # TODO: 4. 从概率分布中采样一个词元。
        #    - 使用 `torch.multinomial` 从 `probs` 中随机抽取 1 个样本。
        next_token_idx = ...
        
        # TODO: 5. 将新生成的词元拼接到上下文中，为下一次迭代做准备。
        #    - 使用 `torch.cat` 将 `context` 和 `next_token_idx` 沿 `dim=1` 拼接。
        context = ...
        
        token_id = next_token_idx.item()
        if token_id == tokenizer.eos_token_id:
            break
            
        new_char = tokenizer.decode([token_id])
        print(new_char, end='', flush=True)
    # --- END OF STUDENT MODIFICATION (GENERATION LOOP) ---
    
    print("\n\n--- End of Generation ---")