# vision_transformer/vit.py
#* OK
import torch
import torch.nn as nn
from typing import Optional
from transformer_from_scratch.model import TransformerEncoder

class PatchEmbedding(nn.Module):
    """
    Converts a batch of images into a sequence of flattened patch embeddings.
    
    This is achieved efficiently using a single convolutional layer.
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        """
        Args:
            img_size (int): Size of the input image (height or width).
            patch_size (int): Size of each patch.
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            d_model (int): The dimensionality of the embedding space (and the Transformer).
        """
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size.")
            
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # A convolutional layer that acts as the patch projection.
        # Kernel size and stride are equal to patch_size, so the kernel moves
        # over the image in non-overlapping steps, processing one patch at a time.
        self.projection = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input images, shape [B, C, H, W]

        Returns:
            torch.Tensor: Sequence of patch embeddings, shape [B, N, D]
                          where N is the number of patches and D is d_model.
        """
        # Project images into patches: [B, C, H, W] -> [B, D, H/P, W/P]
        x = self.projection(x)
        
        # Flatten the spatial dimensions: [B, D, H/P, W/P] -> [B, D, N]
        x = x.flatten(2)
        
        # Transpose to get the sequence dimension first: [B, D, N] -> [B, N, D]
        x = x.transpose(1, 2)
        
        return x

class ViT(nn.Module):
    """
    The Vision Transformer (ViT) model.
    
    This model can be used for two purposes:
    1. Image Classification: If `num_classes` is provided, it acts as a standard ViT classifier.
    2. Feature Extraction: If `num_classes` is None, it acts as a vision encoder,
       outputting the sequence of patch embeddings for use in downstream tasks like MLLMs.
    """
    def __init__(self, 
                 img_size: int, 
                 patch_size: int, 
                 in_channels: int, 
                 d_model: int, 
                 num_layers: int, 
                 n_heads: int, 
                 d_ff: int, 
                 num_classes: Optional[int] = None,
                 dropout: float = 0.1):
        """
        Args:
            img_size (int): Size of the input image (height or width).
            patch_size (int): Size of each patch.
            in_channels (int): Number of input channels.
            d_model (int): The dimensionality of the embedding space.
            num_layers (int): Number of Transformer Encoder layers.
            n_heads (int): Number of attention heads.
            d_ff (int): The dimensionality of the feed-forward layer.
            num_classes (Optional[int]): The number of output classes for classification.
                                         If set to None, the model will not have a classification
                                         head and will output feature embeddings. Defaults to None.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embedding.n_patches
        
        # [CLS] token is a learnable parameter that will aggregate global image features.
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional embeddings for each patch and the [CLS] token.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = TransformerEncoder(num_layers, d_model, n_heads, d_ff, dropout)
        
        # The classification head is only created if num_classes is specified.
        if num_classes is not None:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, num_classes)
            )
        else:
            # In feature extraction mode, there is no classification head.
            self.mlp_head = None

    # --- START OF STUDENT MODIFICATION ---

    def forward_features(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Processes images and returns the sequence of feature embeddings from the encoder.
        This is the core feature extraction logic.

        Args:
            x (torch.Tensor): Input images, shape [B, C, H, W].
            mask (torch.Tensor, optional): Not typically used in ViT, but included for API consistency.

        Returns:
            torch.Tensor: The full sequence of token embeddings from the encoder, shape [B, N+1, D].
        """
        # 获取批量大小
        batch_size = x.shape[0]

        # TODO: 实现 ViT 的特征提取流程
        # 1. 将输入图像 x 转换为块嵌入 (patch embeddings)
        #    - 使用 self.patch_embedding
        #    - Shape: [B, C, H, W] -> [B, N, D]
        x = self.patch_embedding(x)

        # 2. 准备 [CLS] 词元并将其拼接到块嵌入序列的开头
        #    a. 将 self.cls_token [1, 1, D] 扩展以匹配批量大小 (使用 .expand() 方法) #* 不要用 .repeat()，因为 .expand() 不会复制内存，是 ViT 常用写法
        #    b. 使用 torch.cat() 将 cls_tokens 与 x 沿序列维度 (dim=1) 拼接
        #    - Shape: [B, N, D] -> [B, N+1, D]
        cls_tokens = self.cls_token.expand(batch_size, 1, x.shape[2])
        x = torch.cat([cls_tokens, x], dim=1) #* 不能直接写+，因为在对应维度拼接; 注意先后


        # 3. 添加位置编码 (positional embeddings)
        #    - 直接将 self.pos_embedding 加到 x 上
        x = x + self.pos_embedding

        # 4. 对结果应用 dropout
        #    - 使用 self.dropout
        x = self.dropout(x)

        # 5. 将处理后的序列输入到 Transformer 编码器中
        #    - 使用 self.encoder
        #    - 注意：为保持API一致性，传入 src_mask，但在标准ViT中通常为None
        #    - Shape 保持 [B, N+1, D]
        x = self.encoder(x, mask)

        return x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the full forward pass of the ViT model.
        This method should orchestrate the call to `forward_features` and the subsequent
        classification head (if it exists).

        Args:
            x (torch.Tensor): Input images, shape [B, C, H, W].
            mask (torch.Tensor, optional): Not typically used in ViT, but included for API consistency.

        Returns:
            torch.Tensor: 
            - If in classification mode (num_classes is not None), returns logits for each class,
              shape [B, num_classes].
            - If in feature extraction mode (num_classes is None), returns the full sequence of
              token embeddings from the encoder, shape [B, N+1, D].
        """
        # TODO: 实现完整的 ViT 前向传播
        # 1. 调用 self.forward_features() 获取编码器的输出特征
        #    - Shape: [B, N+1, D]
        features = self.forward_features(x)

        # 2. 根据模型是用于分类还是特征提取，执行不同操作
        #    - 使用 `if self.mlp_head is not None:` 进行判断
        if self.mlp_head is not None:
            # --- 分类模式 ---
            # a. 从特征序列中提取 [CLS] 词元的输出 (它位于序列的第一个位置)
            #    - Shape: [B, D]
            cls_output = features[:, 0, :] #* 不需要 squeeze：数字索引（0）会移除这个维度 → 得到 [B, D]；切片（0:1）会保留维度 → 得到 [B, 1, D]。
            
            # b. 将 [CLS] 词元的输出送入分类头 (self.mlp_head) 以获得 logits
            #    - Shape: [B, num_classes]
            logits = self.mlp_head(cls_output)
            return logits
        else:
            # --- 特征提取模式 ---
            # 直接返回编码器的完整输出序列
            return features #* 不用再写 self.forward_features(x)，还得 forward 一次
            
    # --- END OF STUDENT MODIFICATION ---