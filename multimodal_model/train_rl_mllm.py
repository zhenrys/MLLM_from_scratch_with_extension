import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
import random
import torchvision.transforms as transforms

from vision_transformer.vit import ViT
from language_model.llm import GPTModel
from language_model.tokenizer import CharacterTokenizer
from .mllm import MLLM
from .connector import Connector
from datasets.Flickr8k import Flickr8kDataset
from utils.training_utils import get_device


# ============================================================
#  Reward Function: Bigram Overlap  (BLEU-2 style simplified)
# ============================================================
def tokenize_caption(text):
    return list(text.lower().strip())  # char-level, consistent with tokenizer


def bigram_overlap(pred, ref):
    pred_tokens = tokenize_caption(pred)
    ref_tokens = tokenize_caption(ref)

    if len(pred_tokens) < 2 or len(ref_tokens) < 2:
        return 0.0

    def bigrams(tokens):
        return [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]

    pred_bi = bigrams(pred_tokens)
    ref_bi = bigrams(ref_tokens)

    overlap = sum([1 for bg in pred_bi if bg in ref_bi])
    precision = overlap / len(pred_bi)
    return precision


def batch_reward(pred_list, ref_list):
    rewards = [bigram_overlap(p, r) for p, r in zip(pred_list, ref_list)]
    return torch.tensor(rewards, dtype=torch.float32)


# ============================================================
#     RL Fine-Tuning Main Function
# ============================================================
def rl_finetune(config):
    """
    SCST-style RL fine-tuning for MLLM.
    """
    # ---------------------------
    # 0. Environment Setup
    # ---------------------------
    device = get_device(config["training"]["device"])
    train_cfg = config["training"]
    model_cfg = config["model"]
    paths_cfg = config["paths"]

    print("="*50)
    print("INFO: Starting RL Fine-tuning for MLLM (SCST).")
    print("="*50)

    # ---------------------------
    # 1. Load Tokenizer
    # ---------------------------
    tokenizer = CharacterTokenizer(corpus=None)
    tokenizer.load_vocab(paths_cfg["tokenizer_save_path"])
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.get_pad_token_id()

    # ---------------------------
    # 2. Initialize Models
    # ---------------------------
    vision_encoder = ViT(
        img_size=model_cfg["vision_encoder"]["image_size"],
        patch_size=model_cfg["vision_encoder"]["patch_size"],
        in_channels=model_cfg["vision_encoder"]["in_channels"],
        d_model=model_cfg["vision_encoder"]["vision_dim"],
        num_layers=model_cfg["vision_encoder"]["n_layers"],
        n_heads=model_cfg["vision_encoder"]["n_heads"],
        d_ff=model_cfg["vision_encoder"]["d_ff"],
        dropout=model_cfg["vision_encoder"]["dropout"],
        num_classes=model_cfg["vision_encoder"]["num_classes"]
    ).to(device)

    language_model = GPTModel(
        vocab_size=vocab_size,
        d_model=model_cfg["language_model"]["language_dim"],
        num_layers=model_cfg["language_model"]["n_layers"],
        n_heads=model_cfg["language_model"]["n_heads"],
        d_ff=model_cfg["language_model"]["d_ff"],
        max_len=model_cfg["language_model"]["max_len"],
        dropout=model_cfg["language_model"]["dropout"]
    ).to(device)

    connector = Connector(
        vision_dim=model_cfg["connector"]["vision_dim"],
        language_dim=model_cfg["connector"]["language_dim"],
        connector_type=model_cfg["connector"]["type"],
        hidden_dim=model_cfg["connector"]["hidden_dim"]
    ).to(device)

    mllm = MLLM(vision_encoder, language_model, connector, tokenizer).to(device)

    # Load pretrained SFT checkpoint
    print(f"Loading SFT checkpoint: {paths_cfg['best_model_save_path']}")
    mllm.load_state_dict(torch.load(paths_cfg["best_model_save_path"], map_location=device))

    mllm.freeze_parameters(train_cfg["freeze_vit"], train_cfg["freeze_llm"])

    # ---------------------------
    # 3. Prepare Dataset
    # ---------------------------
    transform = torch.nn.Sequential()  # You already use transforms in SFT training

    transform = transforms.Compose([
        transforms.Resize((model_cfg['vision_encoder']['image_size'], model_cfg['vision_encoder']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = Flickr8kDataset(
        root=config["data"]["data_root"],
        captions_file="captions.txt",
        transform=transform,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        full_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch], dim=0),
            pad_sequence([b[1] for b in batch], batch_first=True, padding_value=pad_token_id)
        )
    )

    # ---------------------------
    # 4. Optimizer
    # ---------------------------
    optimizer = optim.Adam(
        [p for p in mllm.parameters() if p.requires_grad],
        lr=train_cfg["learning_rate"]
    )

    # ---------------------------
    # 5. RL Training Loop (SCST)
    # ---------------------------
    rl_losses_per_epoch = []
    ce_losses_per_epoch = []
    adv_values_per_epoch = []
    
    for epoch in range(train_cfg["epochs"]):
        mllm.train()
        pbar = tqdm(train_loader, desc=f"RL Epoch {epoch+1}/{train_cfg['epochs']}")

        total_rl_loss = 0.0
        total_ce_loss = 0.0
        total_adv = 0.0
        total_steps = 0

        for images, captions in pbar:
            images = images.to(device)
            captions = captions.to(device)

            # Prompt = <sos>
            sos = tokenizer.get_sos_token_id()
            prompt_tokens = torch.full((images.size(0), 1), sos, dtype=torch.long, device=device)

            # Convert reference captions to text
            ref_texts = [tokenizer.decode(cap.tolist()) for cap in captions]

            # ============ 1) SAMPLE from policy ===============
            sampled_ids, sampled_logprobs = mllm.sample_with_logprobs(
                images, prompt_tokens, 
                max_new_tokens=train_cfg["max_new_tokens"], 
                temperature=train_cfg["temperature"]
            )
            sampled_texts = [tokenizer.decode(ids.tolist()) for ids in sampled_ids]

            # ============ 2) GREEDY BASELINE ===============
            with torch.no_grad():
                greedy_ids, _ = mllm.greedy_with_logprobs(
                    images, prompt_tokens, 
                    max_new_tokens=train_cfg["max_new_tokens"]
                )
                greedy_texts = [tokenizer.decode(ids.tolist()) for ids in greedy_ids]

            # ============ 3) COMPUTE REWARD ===============
            r_sample = batch_reward(sampled_texts, ref_texts).to(device)
            r_greedy = batch_reward(greedy_texts, ref_texts).to(device)
            advantage = (r_sample - r_greedy).detach()  # [B]

            # ============ 4) RL LOSS ===============
            # mask out PAD positions in logprobs
            pad_mask = (sampled_ids != pad_token_id).float()[:, 1:]  # ignore prompt position
            logprobs = sampled_logprobs * pad_mask  # [B, T]

            rl_loss = -(advantage.unsqueeze(1) * logprobs).sum(dim=1).mean()

            # ============ 5) Mixed CE loss for stability ============
            lambda_rl = train_cfg.get("lambda_rl", 1.0)
            lambda_ce = train_cfg.get("lambda_ce", 0.0)

            ce_loss = torch.tensor(0.0, device=device)
            if lambda_ce > 0:
                model_in = captions[:, :-1]
                targets = captions[:, 1:]

                logits, num_vis = mllm(images, model_in)
                #* labels = torch.full_like(logits[:, :, 0], pad_token_id, device=device)
                # 正确写法：手动指定 dtype=torch.long
                labels = torch.full(
                    (logits.size(0), logits.size(1)),   # [B, T_total]
                    fill_value=pad_token_id,
                    dtype=torch.long,                   # 关键：long 类型
                    device=device,
                )
                labels[:, num_vis : num_vis + targets.size(1)] = targets

                ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
                ce_loss = ce_loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            loss = lambda_rl * rl_loss + lambda_ce * ce_loss

            # ============ 6) Optimize ===============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                "rl_loss": rl_loss.item(),
                "ce_loss": ce_loss.item() if lambda_ce > 0 else 0.0,
                "adv": advantage.mean().item()
            })

            total_rl_loss += rl_loss.item()
            total_ce_loss += ce_loss.item() if lambda_ce > 0 else 0.0
            total_adv += advantage.mean().item()
            total_steps += 1

        # Save checkpoint each epoch
        save_path = paths_cfg["rl_save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(mllm.state_dict(), save_path)
        
        # Epoch average stats
        rl_losses_per_epoch.append(total_rl_loss / total_steps)
        ce_losses_per_epoch.append(total_ce_loss / total_steps)
        adv_values_per_epoch.append(total_adv / total_steps)

        print(f"Saved RL model to {save_path}")

    # =============================
    # Plot training curves
    # =============================
    import matplotlib.pyplot as plt

    epochs = list(range(1, train_cfg["epochs"] + 1))
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, rl_losses_per_epoch, label="RL Loss", marker='o')
    plt.plot(epochs, ce_losses_per_epoch, label="CE Loss", marker='o')
    plt.plot(epochs, adv_values_per_epoch, label="Advantage", marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Loss / Advantage")
    plt.title("RL Fine-tuning Training Curves")
    plt.legend()
    plt.grid(True)

    curve_path = os.path.join(paths_cfg["output_dir"], "rl_loss_curve.png")
    plt.savefig(curve_path)
    print(f"RL training curve saved to {curve_path}")

    plt.close()

    
    print("RL Fine-tuning Completed.")
