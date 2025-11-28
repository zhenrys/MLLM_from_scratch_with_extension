# language_model/train_llm.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets.tinyshakespeare import TinyShakespeareDataset
from language_model.llm import GPTModel
from utils.training_utils import set_seed

@torch.no_grad()
def generate_sample_text(model: GPTModel, tokenizer, device: torch.device, 
                         start_string: str, max_new_tokens: int, block_size: int):
    # 此函数无需修改
    model.eval()
    start_ids = tokenizer.encode(start_string)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits = model(x_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        if idx_next.item() == tokenizer.eos_token_id:
            break
        x = torch.cat((x, idx_next), dim=1)
    generated_ids = x[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    model.train()
    return generated_text

def train(config: dict):
    data_cfg = config['data_params']
    model_cfg = config['model_params']
    train_cfg = config['training_params']
    
    set_seed(42)
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    full_dataset = TinyShakespeareDataset(root=data_cfg['data_dir'], block_size=data_cfg['block_size'], download=True)
    model_cfg['vocab_size'] = full_dataset.tokenizer.get_vocab_size()
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], num_workers=train_cfg['num_workers'])

    model = GPTModel(vocab_size=model_cfg['vocab_size'], d_model=model_cfg['d_model'], num_layers=model_cfg['num_layers'], n_heads=model_cfg['n_heads'], d_ff=model_cfg['d_ff'], max_len=data_cfg['block_size'], dropout=model_cfg['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    
    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(train_cfg['model_save_path']), exist_ok=True)

    train_losses = []
    val_losses = []
    
    for epoch in range(train_cfg['num_epochs']):
        model.train()
        train_loss = 0
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']} [Train]")
        
        # --- START OF STUDENT MODIFICATION (TRAINING LOOP) ---
        for x, y in train_loop:
            x, y = x.to(device), y.to(device)
            
            # TODO: 1. 清空优化器的梯度
            optimizer.zero_grad()

            # TODO: 2. 模型前向传播，得到 logits
            logits = model(x)
            
            # TODO: 3. 计算损失。
            #    - `F.cross_entropy` 需要的 logits 形状是 (N, C)，目标 y 的形状是 (N)。
            #    - 当前 logits 的形状是 (B, T, C)，y 的形状是 (B, T)。
            #    -你需要使用 `.view()` 或 `.reshape()` 来调整它们的形状。
            #    - B: batch_size, T: sequence_length, C: vocab_size
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T)) #! 交叉熵计算不能三维输入, 所以用 view
            
            # TODO: 4. 反向传播
            loss.backward()
            
            # TODO: 5. 更新模型权重
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        # --- END OF STUDENT MODIFICATION (TRAINING LOOP) ---

        avg_train_loss = train_loss / len(train_loader)
        
        if (epoch + 1) % train_cfg['eval_interval'] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                eval_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']} [Eval]")
                for x, y in eval_loop:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    # --- START OF STUDENT MODIFICATION (EVALUATION) ---
                    # TODO: 计算验证集上的损失。
                    #   - 这里的逻辑与训练循环中的损失计算完全相同。
                    B, T, C = logits.shape
                    loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
                    # --- END OF STUDENT MODIFICATION (EVALUATION) ---
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"\nEpoch {epoch+1}/{train_cfg['num_epochs']} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            print("\n--- Generating Sample Text ---")
            start_prompt = "O Romeo, Romeo! wherefore art thou"
            generated_text = generate_sample_text(model=model, tokenizer=full_dataset.tokenizer, device=device, start_string=start_prompt, max_new_tokens=150, block_size=data_cfg['block_size'])
            print(generated_text)
            print("--- End of Sample ---\n")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), train_cfg['model_save_path'])
                print(f"New best model saved to {train_cfg['model_save_path']} with Val Loss: {avg_val_loss:.4f}")
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GPT LM Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("gpt_training_curve.png")
    print("Saved training curve to gpt_training_curve.png")

    
    print("Training finished.")