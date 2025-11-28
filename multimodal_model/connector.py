# multimodal_model/connector.py

import torch.nn as nn

class Connector(nn.Module):
    """
    将视觉特征从 ViT 的表示空间映射到 LLM 的表示空间。
    
    [ 任务 ]
    1. 在 __init__ 方法中，完成 MLP 网络的构建。
    2. 在 forward 方法中，实现特征的前向传播。
    """
    def __init__(self, vision_dim: int, language_dim: int, hidden_dim: int = None, connector_type: str = "mlp"):
        """
        Args:
            vision_dim (int): ViT 输出的特征维度 (d_model of ViT).
            language_dim (int): LLM 输入的嵌入维度 (d_model of LLM).
            hidden_dim (int, optional): 如果使用MLP，其中间隐藏层的维度. Defaults to None.
            connector_type (str, optional): 连接器类型, 'linear' or 'mlp'. Defaults to "mlp".
        """
        super().__init__()
        self.connector_type = connector_type

        if connector_type == "linear":
            self.model = nn.Linear(vision_dim, language_dim)
        elif connector_type == "mlp":
            if hidden_dim is None:
                hidden_dim = (vision_dim + language_dim) // 2
            
            # --- START OF STUDENT TASK 1 ---
            # TODO: 构建一个 MLP (多层感知机) 模型。
            # 这是一个简单的序列模型，用于将视觉特征“翻译”成语言特征。
            # 结构应为:
            # 1. 一个线性层 (Linear)，将维度从 vision_dim 映射到 hidden_dim。
            # 2. 一个激活函数 (GELU)，GELU 在 Transformer 模型中很常用。
            # 3. 另一个线性层 (Linear)，将维度从 hidden_dim 映射到最终的 language_dim。
            # 提示: 使用 nn.Sequential 将这些层组合起来。
            
            self.model = nn.Sequential(
                nn.Linear(vision_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, language_dim)
            )
            # --- END OF STUDENT TASK 1 ---
            
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 来自 ViT 的视觉特征, shape [B, N, D_vision]
        
        Returns:
            torch.Tensor: 映射到语言空间的嵌入, shape [B, N, D_language]
        """
        # --- START OF STUDENT TASK 2 ---
        # TODO: 实现前向传播。
        # 将输入 x 通过上面定义的 self.model 进行处理并返回结果。
        
        # YOUR CODE HERE
        #* model 是 sequential 封装的，必须满足第一个模块linear（in，out） 的输入输出
        #* Linear 的输入 shape 必须是：[..., in_features]。输出 shape 是：[..., out_features]
        
        return self.model(x)
        # --- END OF STUDENT TASK 2 ---