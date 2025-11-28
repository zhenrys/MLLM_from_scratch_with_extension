# transformer_from_scratch/blocks.py

import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward

class EncoderBlock(nn.Module):
    """
    Transformer Encoder 的基本构建块。
    包含一个多头自注意力层和一个位置前馈网络。
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # --- YOUR CODE HERE ---
        #* OK: 实例化多头自注意力层和位置前馈网络层。
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # --- END YOUR CODE ---

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): EncoderBlock的输入, shape [B, L_src, d_model]
            src_mask (torch.Tensor): 源序列的掩码

        Returns:
            torch.Tensor: EncoderBlock的输出, shape [B, L_src, d_model]
        """
        # --- YOUR CODE HERE ---
        #* OK : 实现 EncoderBlock 的前向传播
        
        # 1. 通过多头自注意力层。注意 Q, K, V 都来自 src。
        #    Add & Norm 已在 MultiHeadAttention 内部实现。
        src = self.self_attn(src, src, src, mask = src_mask)
        
        # 2. 通过位置前馈网络。
        #    Add & Norm 已在 PositionwiseFeedForward 内部实现。
        src = self.feed_forward(src)
        
        return src
        # --- END YOUR CODE ---

class DecoderBlock(nn.Module):
    """
    Transformer Decoder 的基本构建块。
    包含一个掩码多头自注意力层、一个多头交叉注意力层和一个位置前馈网络。
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # --- YOUR CODE HERE ---
        # OK: 实例化三个核心组件
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)   # 用于目标序列的自注意力
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)   # 用于 encoder 和 decoder 之间的交叉注意力
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout) # 位置前馈网络
        # --- END YOUR CODE ---

    def forward(self, tgt: torch.Tensor, enc_src: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor], src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            tgt (torch.Tensor): DecoderBlock的输入, shape [B, L_tgt, d_model]
            enc_src (torch.Tensor | None): Encoder的输出, shape [B, L_src, d_model]。
            tgt_mask (torch.Tensor | None): 目标序列的掩码 (用于自注意力)
            src_mask (torch.Tensor | None): 源序列的掩码 (用于交叉注意力)

        Returns:
            torch.Tensor: DecoderBlock的输出, shape [B, L_tgt, d_model]
        """
        # --- YOUR CODE HERE ---
        # TODO: 实现 DecoderBlock 的前向传播

        # 1. 掩码多头自注意力。Q, K, V 都来自 tgt，使用 tgt_mask。
        tgt = self.self_attn(tgt, tgt, tgt, mask = tgt_mask) #* 此处使用 mask！
        
        # 只有在 enc_src (Encoder的输出) 被提供时，才执行交叉注意力。
        if enc_src is not None:
            # 2. 多头交叉注意力。Q 来自上一步的输出，K 和 V 来自 encoder 的输出 enc_src。
            #    使用 src_mask。
            tgt = self.cross_attn(tgt, enc_src, enc_src, mask=src_mask)
        
        # 3. 位置前馈网络。
        tgt =  self.feed_forward(tgt)
        
        return tgt
        # --- END YOUR CODE ---
