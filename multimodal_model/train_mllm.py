# multimodal_model/train_mllm.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
import pandas as pd
import torchvision.transforms as transforms
import random

from utils.training_utils import get_device
from vision_transformer.vit import ViT
from language_model.llm import GPTModel
from language_model.tokenizer import CharacterTokenizer
from .connector import Connector
from .mllm import MLLM
from datasets.Flickr8k import Flickr8kDataset

# ... (helper functions remain the same) ...
def generate_caption_for_sample(mllm_model, tokenizer, image, device, max_len=50):
    mllm_model.eval()
    image = image.unsqueeze(0).to(device)
    start_prompt = '' 
    with torch.no_grad():
        generated_caption = mllm_model.generate(
            image=image, prompt=start_prompt, max_new_tokens=max_len, temperature=0.8, top_k=10)
    return generated_caption

def create_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def build_tokenizer(corpus_path, save_path):
    if os.path.exists(save_path):
        print(f"Loading existing tokenizer from {save_path}")
        tokenizer = CharacterTokenizer(corpus=None); tokenizer.load_vocab(save_path)
    else:
        print(f"Building tokenizer from {corpus_path}...")
        df = pd.read_csv(corpus_path); corpus = "\n".join(df['caption'].tolist())
        tokenizer = CharacterTokenizer(corpus); tokenizer.save_vocab(save_path)
        print(f"Tokenizer saved to {save_path}")
    return tokenizer


def train(config):
    """
    The main training function for the Multimodal Large Language Model (MLLM).
    
    #TODO [ 任务 ]
    1. 在训练循环中，正确地准备语言模型的输入 (model_input_text) 和目标 (targets)。
    2. 在训练循环中，实现最关键的损失计算逻辑，确保只计算文本部分的损失。
    3. 在评估循环中，应用与训练循环中相同的损失计算逻辑。
    """
    # --- 0. Setup ---
    device = get_device(config['training']['device'])
    train_cfg = config['training']; paths_cfg = config['paths']; model_cfg = config['model']
    print("="*50); print("INFO: Using robust Flickr8kDataset implementation."); print("="*50)

    # --- 1. Build or Load Tokenizer ---
    tokenizer = build_tokenizer(paths_cfg['captions_corpus_path'], paths_cfg['tokenizer_save_path'])
    vocab_size = tokenizer.get_vocab_size(); pad_token_id = tokenizer.get_pad_token_id()

    # --- 2. Initialize Models ---
    #* Args:
        # img_size (int): Size of the input image (height or width).
        # patch_size (int): Size of each patch.
        # in_channels (int): Number of input channels.
        # d_model (int): The dimensionality of the embedding space.
        # num_layers (int): Number of Transformer Encoder layers.
        # n_heads (int): Number of attention heads.
        # d_ff (int): The dimensionality of the feed-forward layer.
        # num_classes (Optional[int]): The number of output classes for classification.
        #                                 If set to None, the model will not have a classification
        #                                 head and will output feature embeddings. Defaults to None.
        # dropout (float): Dropout rate.
    vision_encoder = ViT(
        img_size=model_cfg['vision_encoder']["image_size"], 
        patch_size=model_cfg['vision_encoder']["patch_size"], 
        in_channels=model_cfg['vision_encoder']["in_channels"], 
        d_model=model_cfg['vision_encoder']["vision_dim"], 
        num_layers=model_cfg['vision_encoder']["n_layers"], 
        n_heads=model_cfg['vision_encoder']["n_heads"], 
        d_ff=model_cfg['vision_encoder']["d_ff"], 
        dropout=model_cfg['vision_encoder']["dropout"],
        num_classes=model_cfg['vision_encoder']["num_classes"] #* 通常可以直接num_classes=None
    ).to(device) # Simplified for brevity
    
    #* def __init__(self, vocab_size: int, d_model: int, num_layers: int, n_heads: int, d_ff: int, max_len: int, dropout: float):
    language_model = GPTModel(
        vocab_size=vocab_size,
        d_model=model_cfg['language_model']["language_dim"],
        num_layers=model_cfg['language_model']["n_layers"],
        n_heads=model_cfg['language_model']["n_heads"],
        d_ff=model_cfg['language_model']["d_ff"],
        max_len=model_cfg['language_model']["max_len"],
        dropout=model_cfg['language_model']["dropout"],
    ).to(device) # Simplified for brevity
    
    #* Args:
    #     vision_dim (int): ViT 输出的特征维度 (d_model of ViT).
    #     language_dim (int): LLM 输入的嵌入维度 (d_model of LLM).
    #     hidden_dim (int, optional): 如果使用MLP，其中间隐藏层的维度. Defaults to None.
    #     connector_type (str, optional): 连接器类型, 'linear' or 'mlp'. Defaults to "mlp".
    connector = Connector(
        vision_dim=model_cfg['connector']['vision_dim'],
        language_dim=model_cfg['connector']['language_dim'],
        connector_type=model_cfg['connector']['type'],
        hidden_dim=model_cfg['connector']['hidden_dim'],
    ).to(device) # Simplified for brevity
    
    
    mllm = MLLM(vision_encoder, language_model, connector, tokenizer).to(device)
    mllm.freeze_parameters(train_cfg['freeze_vit'], train_cfg['freeze_llm'])

    # --- 3. Prepare Data ---
    transform = create_transform(model_cfg['vision_encoder']['image_size'])
    full_dataset = Flickr8kDataset(root=config['data']['data_root'], captions_file="captions.txt", transform=transform, tokenizer=tokenizer)
    train_size = int(0.9 * len(full_dataset)); val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    
    def collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images, 0)
        captions_padded = pad_sequence(captions, batch_first=True, padding_value=pad_token_id)
        return images, captions_padded

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'], collate_fn=collate_fn)
    sample_img, sample_gt_tensor = random.choice(val_dataset)
    print("A fixed sample image has been chosen for generation during training.")

    # --- 4. Setup Optimizer and Loss ---
    optimizer = optim.Adam(mllm.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    print("Starting MLLM training...")
     
    train_losses = [] # 绘图
    val_losses = [] # 绘图
    
    for epoch in range(train_cfg['epochs']):
        mllm.train()
        total_train_loss = 0
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{train_cfg['epochs']} [Train]")
        for images, captions in train_loop:
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # --- START OF STUDENT TASK 1 ---
            # TODO: 为语言模型准备输入 (model_input_text) 和目标 (targets)。
            # 对于一个自回归模型，输入是序列的前n-1个token，目标是序列的后n-1个token。
            # 例如，如果完整序列是 [<sos>, "a", "cat", <eos>]
            # 输入应该是: [<sos>, "a", "cat"]
            # 目标应该是: ["a", "cat", <eos>]
            # 提示: 使用 Python 的切片操作。`captions` 的形状是 [B, SeqLen]。
            
            model_input_text = captions[:, :-1] #* 错误示范 = captions[:-1]，这会让 batch 维度消失
            targets = captions[:, 1:]          # YOUR CODE HERE
            # --- END OF STUDENT TASK 1 ---
            
            logits, num_visual_tokens = mllm(images, model_input_text)
            
            # --- START OF STUDENT TASK 2 (CRITICAL) ---
            # TODO: 计算损失。这是 MLLM 训练中最关键的一步。
            #! 最最最关键一步——————制作 label！！！
            # 问题: `logits` 的序列长度是 (视觉token数量 + 文本token数量)。
            # 但我们只想让模型学会预测文本部分，而不关心它对视觉token的预测。
            # 因此，我们需要在计算损失时“忽略”掉对应视觉token位置的预测。
            #
            # 解决方案: 使用 `criterion.ignore_index` (它等于 pad_token_id)。
            # `CrossEntropyLoss` 会自动忽略 `labels` 中值为 `ignore_index` 的位置。
            #
            # 步骤:
            # 1. 创建一个 `labels` 张量，其形状与 `logits` 的前两个维度相同 ([B, SeqLen])。
            #    用 `criterion.ignore_index` 将其完全填充。
            #    提示: 使用 `torch.full()`。
            # 2. 计算出文本 `targets` 在 `labels` 张量中应该被放置的起始和结束索引。
            #    文本内容紧跟在视觉token之后。`num_visual_tokens` 变量会告诉你视觉token的数量。
            # 3. 将真实的 `targets` "粘贴" 到 `labels` 张量的正确位置上。
            #    这样 `labels` 就变成了 `[ignore, ignore, ..., target_1, target_2, ...]` 的形式。
            # 4. 最后，计算损失。将 `logits` 和 `labels` 都展平 (view(-1)) 后传入 `criterion`。
            
            # 步骤 1: 创建一个填满 ignore_index 的 labels 张量
            labels = torch.full(
                (logits.shape[0],logits.shape[1]),
                fill_value=criterion.ignore_index, #* 这样使用
                device=logits.device
            ) # YOUR CODE HERE

            # 步骤 2: 确定 targets 的起止位置
            label_start_idx = num_visual_tokens # YOUR CODE HERE
            label_end_idx = num_visual_tokens + targets.size(1)  # YOUR CODE HERE

            # 步骤 3: 将 targets 放置到 labels 的正确位置
            labels[:, label_start_idx : label_end_idx] = targets
            
            # 步骤 4: 计算损失
            loss = criterion(
                logits.view(-1, logits.size(-1)), #* batch * len个样本，每个样本c个类别。其中 -1 会自动计算为 B * T
                labels.view(-1)
            ) # YOUR CODE HERE
            # --- END OF STUDENT TASK 2 ---
            
            loss.backward(); optimizer.step()
            total_train_loss += loss.item(); train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Evaluation Phase ---
        if (epoch + 1) % train_cfg['eval_interval'] == 0:
            mllm.eval()
            total_val_loss = 0
            with torch.no_grad():
                val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{train_cfg['epochs']} [Eval]")
                for images, captions in val_loop:
                    images, captions = images.to(device), captions.to(device)
                    model_input_text = captions[:, :-1]
                    targets = captions[:, 1:]
                    logits, num_visual_tokens = mllm(images, model_input_text)
                    
                    # --- START OF STUDENT TASK 3 ---
                    # TODO: 在评估阶段应用与训练阶段完全相同的损失计算逻辑。
                    #* 重复上面的步骤 1-4，以计算验证集上的损失。
                    
                    # 步骤 1: 创建一个填满 ignore_index 的 labels 张量
                    labels = torch.full(
                        (logits.shape[0],logits.shape[1]),
                        fill_value=criterion.ignore_index,
                        device=logits.device
                    ) # YOUR CODE HERE

                    # 步骤 2: 确定 targets 的起止位置
                    label_start_idx = num_visual_tokens # YOUR CODE HERE
                    label_end_idx = num_visual_tokens + targets.size(1)  # YOUR CODE HERE

                    # 步骤 3: 将 targets 放置到 labels 的正确位置
                    labels[:, label_start_idx : label_end_idx] = targets
                    
                    # 步骤 4: 计算损失
                    loss = criterion(
                        logits.view(-1, logits.size(-1)), 
                        labels.view(-1)
                    ) # YOUR CODE HERE
                    # --- END OF STUDENT TASK 3 ---
                    
                    total_val_loss += loss.item(); val_loop.set_postfix(loss=loss.item())

            avg_val_loss = total_val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss) # 绘图
            val_losses.append(avg_val_loss) # 绘图
            
            print(f"Epoch {epoch+1}/{train_cfg['epochs']} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            # ... (Sample generation and checkpointing remain the same) ...
            print("\n--- Generating Sample Caption ---")
            gt_caption_full = tokenizer.decode(sample_gt_tensor.tolist())
            gt_caption_clean = (gt_caption_full.replace(tokenizer.sos_token, '').replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip())
            generated_caption = generate_caption_for_sample(mllm_model=mllm, tokenizer=tokenizer, image=sample_img, device=device, max_len=100)
            print(f"  Ground Truth: {gt_caption_clean}"); print(f"  Generated:    {generated_caption}"); print("--- End of Sample ---\n")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(os.path.dirname(paths_cfg['best_model_save_path']), exist_ok=True)
                torch.save(mllm.state_dict(), paths_cfg['best_model_save_path'])
                print(f"New best model saved to {paths_cfg['best_model_save_path']} with Val Loss: {avg_val_loss:.4f}\n")
    import matplotlib.pyplot as plt

    # 创建图表
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Evaluation step (every eval_interval epochs)")
    plt.ylabel("Loss")
    plt.title("MLLM Training Curve")
    plt.legend()

    # 保存到 ckpt 目录
    save_path = os.path.join(paths_cfg['output_dir'], "mllm_loss_curve.png")
    plt.savefig(save_path)
    print(f"Training curve saved to {save_path}")

    plt.close()

    print("Training finished.")