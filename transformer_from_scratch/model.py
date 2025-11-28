# transformer_from_scratch/model.py

import torch
import torch.nn as nn
import math
from .blocks import EncoderBlock, DecoderBlock
from .layers import PositionalEncoding

class TransformerEncoder(nn.Module):
    """
    完整的 Transformer Encoder，由 N 个 EncoderBlock 堆叠而成。
    """
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # --- YOUR CODE HERE ---
        # TODO: 使用 nn.ModuleList 创建一个包含 num_layers 个 EncoderBlock 的列表。
        #* 错误案例 self.layers = nn.ModuleList([num_layers * EncoderBlock])
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # --- END YOUR CODE ---
        self.layer_norm = nn.LayerNorm(d_model) # 最后的层归一化

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # --- YOUR CODE HERE ---
        # TODO: 依次将 src 通过 ModuleList 中的每一个 EncoderBlock。
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.layer_norm(src)
        # --- END YOUR CODE ---

class TransformerDecoder(nn.Module):
    """
    完整的 Transformer Decoder，由 N 个 DecoderBlock 堆叠而成。
    """
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # --- YOUR CODE HERE ---
        # TODO: 使用 nn.ModuleList 创建一个包含 num_layers 个 DecoderBlock 的列表。
        #* 错误案例：self.layers = nn.ModuleList([num_layers * DecoderBlock])
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # --- END YOUR CODE ---
        self.layer_norm = nn.LayerNorm(d_model) # 最后的层归一化

    def forward(self, tgt: torch.Tensor, enc_output: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # --- YOUR CODE HERE ---
        # TODO: 依次将 tgt 通过 ModuleList 中的每一个 DecoderBlock。
        for layer in self.layers:
            tgt = layer(tgt, enc_output, tgt_mask, src_mask) #* decoder 要传入两种 mask
        return self.layer_norm(tgt)
        # --- END YOUR CODE ---

class Transformer(nn.Module):
    """
    一个完整的序列到序列 (Sequence-to-Sequence) Transformer 模型。
    """
    def __init__(self, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int, 
                 d_model: int, 
                 num_layers: int, 
                 n_heads: int, 
                 d_ff: int, 
                 max_len: int = 5000, 
                 dropout: float = 0.1):
        super().__init__()
        
        # --- YOUR CODE HERE ---
        # TODO: 实例化模型的各个组件
        self.src_embedding = nn.Embedding(src_vocab_size, d_model) # 源语言的词嵌入层 #* 注意两个参数
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model) # 目标语言的词嵌入层
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)   # 位置编码器 #* 注意三个参数
        
        self.encoder = TransformerEncoder(num_layers, d_model, n_heads, d_ff, dropout)       # TransformerEncoder #* 是一组 block 而非一个
        self.decoder = TransformerDecoder(num_layers, d_model, n_heads, d_ff, dropout)       # TransformerDecoder
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)      # 最后的线性层，映射到目标词汇表大小
        # --- END YOUR CODE ---
        
        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        处理源序列和目标序列。
        
        Args:
            src (torch.Tensor): 源序列, shape [B, L_src]
            tgt (torch.Tensor): 目标序列, shape [B, L_tgt]
            src_mask (torch.Tensor): 源序列的填充掩码, shape [B, 1, 1, L_src]
            tgt_mask (torch.Tensor): 目标序列的组合掩码 (因果+填充), shape [B, 1, L_tgt, L_tgt]

        Returns:
            torch.Tensor: 模型输出的 logits, shape [B, L_tgt, tgt_vocab_size]
        """
        # --- YOUR CODE HERE ---
        # TODO: 实现完整的 Transformer 前向传播流程

        # 1. 嵌入和位置编码
        #    - 对 src 和 tgt 应用词嵌入。
        #    - 按照论文中的建议，将嵌入结果乘以 sqrt(d_model)。
        #    - 将位置编码应用到经过缩放的嵌入上。#* 不是加上，是复合
        src_processed = self.pos_encoder((self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_processed = self.pos_encoder((self.tgt_embedding(tgt) * math.sqrt(self.d_model)))

        # 2. Encoder
        #    - 将处理后的源序列和源掩码送入 encoder。
        enc_output = self.encoder(src_processed, src_mask)

        # 3. Decoder
        #    - 将处理后的目标序列、encoder的输出、目标掩码和源掩码送入 decoder。
        dec_output = self.decoder(tgt_processed, enc_output, tgt_mask, src_mask)
        
        # 4. 最终线性层
        #    - 将 decoder 的输出送入最后的线性层，得到 logits。
        output = self.fc_out(dec_output)
        
        return output
        # --- END YOUR CODE ---
    
    @staticmethod
    def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        为序列批量创建填充掩码 (padding mask)。
        在非填充 token 的位置为 True。
        
        Args:
            seq (torch.Tensor): 输入序列张量, shape [B, L]
            pad_idx (int): 填充 token 的索引。

        Returns:
            torch.Tensor: 填充掩码, shape [B, 1, 1, L]
        """
        # --- YOUR CODE HERE ---
        # TODO: 创建填充掩码
        # 1. 创建一个布尔张量，其中 `seq` 中不等于 `pad_idx` 的位置为 True。
        #    shape: [B, L]
        mask = (seq != pad_idx) #* seq 与 pad 本身就是张量
        
        # 2. 使用 unsqueeze 增加两个维度，以匹配多头注意力的输入格式 [B, H, L_q, L_k]。
        #    最终形状: [B, 1, 1, L]
        return mask.unsqueeze(1).unsqueeze(2)
        # --- END YOUR CODE ---

    @staticmethod
    def create_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """
        创建因果掩码 (causal / look-ahead mask)。
        在主对角线及以下的位置为 True。
        
        Args:
            size (int): 序列的长度 (L)。
            device (torch.device): 创建张量所在的设备。

        Returns:
            torch.Tensor: 因果掩码, shape [1, 1, L, L]
        """
        # --- YOUR CODE HERE ---
        # TODO: 创建因果掩码
        # 1. 使用 torch.tril 创建一个大小为 (size, size) 的下三角矩阵。
        #    shape: [L, L]
        mask = torch.tril(torch.ones(size, size, device=device)).bool() #! 此处使用 device 防止 mask 在 CPU 上
        #* 使用 bool 将 mask 变为 bool 型
        # 2. 使用 unsqueeze 增加两个维度，以用于广播。
        #    最终形状: [1, 1, L, L]
        return mask.unsqueeze(0).unsqueeze(0)
        # --- END YOUR CODE ---