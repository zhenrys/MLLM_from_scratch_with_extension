# utils/training_utils.py

import torch
import numpy as np
import random
import os

def set_seed(seed: int):
    """
    为CPU、GPU和各种随机数库设置随机种子，以确保实验的可复现性。

    Args:
        seed (int): 要设置的种子数值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 以下两行通常用于确保在使用CUDA时的完全可复现性，
        # 但可能会对性能产生轻微影响。如果需要精确的逐位复现，请取消注释。
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def get_device(device_str: str = "auto") -> torch.device:
    """
    根据用户请求和系统可用性，确定用于PyTorch操作的设备（GPU或CPU）。

    Args:
        device_str (str, optional): 用户指定的设备字符串，如 "cuda" 或 "cpu"。
                                    默认为 "auto"，将自动选择可用的最佳设备。

    Returns:
        torch.device: PyTorch设备对象。
    """
    # 解释：
    # 1. 检查用户是否请求了 "cuda" 并且系统中确实有可用的CUDA设备。
    if device_str.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU for training.")
    # 2. 如果不满足上述条件（例如，用户请求 "cpu"，或请求 "cuda" 但不可用），则回退到CPU。
    else:
        if device_str.lower() == "cuda":
            print("CUDA was requested but is not available. Falling back to CPU.")
        device = torch.device("cpu")
        print("Using CPU for training.")
    return device

def save_checkpoint(state_dict: dict, filepath: str):
    """
    将模型的状态字典（权重）保存到指定的文件路径。
    如果目标目录不存在，会自动创建。

    Args:
        state_dict (dict): 从 model.state_dict() 获取的模型状态字典。
        filepath (str): 完整的保存文件路径，例如 'outputs/models/mllm_final.pth'。
    """
    # 解释：
    # 1. 使用 os.path.dirname 获取文件所在的目录路径。
    directory = os.path.dirname(filepath)
    # 2. 使用 os.makedirs 创建目录，exist_ok=True 确保如果目录已存在，不会抛出错误。
    os.makedirs(directory, exist_ok=True)
    
    # 3. 使用 torch.save 保存模型的状态字典。
    torch.save(state_dict, filepath)
    
    print(f"Checkpoint saved successfully to {filepath}")