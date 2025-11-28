# transformer_from_scratch/attention.py

import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """
    原生实现缩放点积注意力 (Scaled Dot-Product Attention)。
    
    公式: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): 查询, shape [B, n_heads, L_q, d_k]
            key (torch.Tensor): 键, shape [B, n_heads, L_k, d_k]
            value (torch.Tensor): 值, shape [B, n_heads, L_v, d_v] (L_k=L_v)
            mask (torch.Tensor, optional): 掩码, shape [B, 1, 1, L_k] or [B, 1, L_q, L_k]. Defaults to None.
        #* L_q 为多少个 Q ，L_k 为多少个 key ，d_k为 QK 的统一压缩维度;B是 batchsize
        
        Returns:
            torch.Tensor: 注意力输出, shape [B, n_heads, L_q, d_v]
        """
        # --- YOUR CODE HERE ---
        #* OK: 实现缩放点积注意力机制
        
        # 1. 获取 key 的最后一个维度, 即 d_k。
        d_k = key.size(-1)
        
        # 2. 计算 query 和 key 的转置的点积，得到注意力分数 (scores)。
        #    提示: 使用 torch.matmul()。key需要转置最后两个维度。
        #    scores shape: [B, n_heads, L_q, L_k]
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # 3. 对 scores 进行缩放，除以 sqrt(d_k)。
        scores = scores / math.sqrt(d_k)
        
        # 4. (可选) 应用掩码。如果 mask 不为 None，将 mask 中为 0 (False) 的位置
        #    在 scores 中对应位置替换为一个非常大的负数 (例如 -1e9)。
        #    这使得在 softmax 后，这些位置的概率会趋近于0。
        #    提示: 使用 scores.masked_fill(mask == 0, -1e9)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) #* 将 score 中 mask == 0 的地方替换为很大的负数

        # 5. 对缩放后的 scores 在最后一个维度上应用 softmax，得到注意力权重。
        #    attn_weights shape: [B, n_heads, L_q, L_k]
        attn_weights = torch.softmax(scores, dim=-1) #* 对最后1维度做，保证和为 1
        attn_weights = self.dropout(attn_weights) #!  框架在这里应用 dropout 

        # 6. 将注意力权重与 value 相乘，得到最终的输出。
        #    提示: 使用 torch.matmul()。
        #    output shape: [B, n_heads, L_q, d_v]
        output = torch.matmul(attn_weights, value) #* V是一个方阵

        return output
        # --- END YOUR CODE ---


class MultiHeadAttention(nn.Module):
    """
    原生实现多头注意力 (Multi-Head Attention)。
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): 模型的总维度 (embedding dimension)。
            n_heads (int): 注意力头的数量。
            dropout (float, optional): Dropout概率. Defaults to 0.1.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除" #* d_model 必须能被 n_heads 整除

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads # 每个头的维度

        # --- YOUR CODE HERE ---
        #* OK: 定义 Q, K, V 的线性投影层和最后的输出线性层。
        # 所有层的输入和输出维度都应为 d_model。
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        # --- END YOUR CODE ---
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout) 
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): 查询, shape [B, L_q, d_model]
            key (torch.Tensor): 键, shape [B, L_k, d_model]
            value (torch.Tensor): 值, shape [B, L_v, d_model]
            mask (torch.Tensor, optional): 掩码. Defaults to None.

        Returns:
            torch.Tensor: 多头注意力的输出, shape [B, L_q, d_model]
        """
        batch_size = query.size(0)
        residual = query
        len_q, len_k, len_v = query.size(1), key.size(1), value.size(1)

        # --- YOUR CODE HERE ---
        #* OK: 实现多头注意力的前向传播
        
        # 1. 线性投影: 将输入的 query, key, value 通过对应的线性层。
        #    shape: [B, L, d_model] -> [B, L, d_model]
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        # 2. 拆分成多个头: 将 d_model 维度拆分为 n_heads * d_k。
        #    - 使用 .view() 将形状从 [B, L, d_model] 变为 [B, L, n_heads, d_k]
        #    - 使用 .transpose(1, 2) 交换 n_heads 和 L 维度，以匹配 attention 函数的输入。
        #    - 最终形状: [B, n_heads, L, d_k]
        query = query.view(batch_size, len_q, self.n_heads, self.d_k) #* 将 d_model 维度拆分为 n_heads * 
        query = query.transpose(1, 2) 
        key = key.view(batch_size, len_k, self.n_heads, self.d_k) #* d_k都按 key 分头！
        key = key.transpose(1, 2) 
        value = value.view(batch_size, len_v, self.n_heads, self.d_k) #* L_k == L_v #* d_k都按 key 分头！
        value = value.transpose(1, 2)

        # 3. 计算缩放点积注意力: 调用 self.attention。
        #    context shape: [B, n_heads, L_q, d_k]
        context = self.attention(query, key, value, mask)

        # 4. 合并多个头: 这是第2步的逆操作。
        #    - 使用 .transpose(1, 2) 交换回 n_heads 和 L_q 维度。
        #    - 使用 .contiguous() 来确保内存是连续的。
        #    - 使用 .view() 将形状从 [B, L_q, n_heads, d_k] 合并回 [B, L_q, d_model]。
        context = context.transpose(1, 2).contiguous() #* 张量变形后保证内存连续
        context = context.view(batch_size, len_q, self.d_model)
        
        # 5. 通过最后的线性层。
        output = self.fc(context) # Replace this line
        output = self.dropout(output) #! 此处又添加了 dropout

        # 6. Add & Norm: 添加残差连接并应用 Layer Normalization。
        output = residual + output #* 残差就是 query
        output = self.layer_norm(output)
        
        return output
        # --- END YOUR CODE ---