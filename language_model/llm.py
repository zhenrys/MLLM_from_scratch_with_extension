# language_model/llm.py

import torch
import torch.nn as nn

# 导入您自己实现的 DecoderBlock
from transformer_from_scratch.blocks import DecoderBlock

class GPTModel(nn.Module):
    """
    一个GPT风格（仅解码器）的Transformer模型。
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, n_heads: int, d_ff: int, max_len: int, dropout: float):
        super().__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.drop_emb = nn.Dropout(dropout)
        self.trf_blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_from_embeddings(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        [新方法] 从嵌入向量开始进行前向传播。
        """
        B, T, D = input_embeddings.shape
        if T > self.max_len:
            raise ValueError(f"Input sequence length ({T}) exceeds model's block size ({self.max_len})")

        # --- START OF STUDENT MODIFICATION ---

        # TODO: 1. 创建并添加位置嵌入。
        #    a. 创建一个从 0 到 T-1 的位置索引张量 `pos`。
        #    b. 使用 `self.position_embedding` 将 `pos` 转换为位置嵌入 `pos_emb`。
        #    c. 将 `pos_emb` 添加到 `input_embeddings` 中，得到 `x`。
        #    d. 对 `x` 应用 `self.drop_emb` Dropout。
        pos = torch.arange(0, T, dtype=torch.long, device=input_embeddings.device)
        pos_emb = self.position_embedding(pos)
        x = input_embeddings + pos_emb # 添加位置嵌入 #* 应用了广播性质
        x = self.drop_emb(x) # 应用 Dropout

        # TODO: 2. 创建因果掩码 (causal mask)。
        #    - 这是解码器模型的关键，防止模型在预测当前位置时“看到”未来的信息。
        #    - 创建一个下三角矩阵，形状为 (T, T)，并确保其符合 `DecoderBlock` 的掩码输入格式 (B, 1, T, T)。
        #    - 提示: 使用 `torch.tril` 和 `torch.ones`。
        tgt_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(1) #* 熟练创造下三角矩阵

        # TODO: 3. 依次通过所有 Transformer 块。
        #    - 遍历 `self.trf_blocks` 中的每一个 `block`。
        #    - 将 `x` 和 `tgt_mask` 传入 `block` 进行处理，并更新 `x`。
        #    - 注意 `DecoderBlock` 的签名: `block(tgt=x, enc_src=None, tgt_mask=tgt_mask, src_mask=None)`
        for block in self.trf_blocks:
            x = block(tgt=x, enc_src=None, tgt_mask=tgt_mask, src_mask=None)
        
        # TODO: 4. 应用最终的归一化层和输出头。
        #    a. 将 `x` 传入 `self.final_norm`。
        #    b. 将归一化后的结果传入 `self.out_head` 以获得最终的 logits。
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        # --- END OF STUDENT MODIFICATION ---
        
        return logits

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        [重构] 从词元索引开始的完整前向传播。
        """
        # --- START OF STUDENT MODIFICATION ---
        
        # TODO: 实现从索引到 logits 的完整流程。
        #   1. 使用 `self.token_embedding` 将输入的索引 `idx` 转换为词元嵌入 `tok_emb`。
        #   2. 调用 `self.forward_from_embeddings` 方法，传入 `tok_emb`，以复用核心逻辑并获得 logits。
        tok_emb = self.token_embedding(idx)
        logits = self.forward_from_embeddings(tok_emb)
        
        # --- END OF STUDENT MODIFICATION ---
        
        return logits