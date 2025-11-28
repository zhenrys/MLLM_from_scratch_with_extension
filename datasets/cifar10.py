# datasets/cifar10.py

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from PIL.Image import Image

class CIFAR10Dataset(Dataset):
    """
    A wrapper for the torchvision CIFAR-10 dataset.
    """
    def __init__(self, root: str = "data", train: bool = True, transform=None, download: bool = True):
        self.transform = transform
        self.cifar10 = CIFAR10(root=root, train=train, download=download)
        self.classes = self.cifar10.classes

    def __len__(self) -> int:
        return len(self.cifar10)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image, int]:
        """
        Fetches the sample at the given index.
        """
        # The torchvision dataset returns a (PIL Image, label) tuple
        image, label = self.cifar10[idx]

        # --- START OF STUDENT MODIFICATION ---
        
        # TODO: 应用图像变换
        # 如果 self.transform 不为 None，则需要将它应用到 image 上。
        # 这是 PyTorch 数据集处理的标准流程。
        if self.transform:
            image = self.transform(image)
        
        # --- END OF STUDENT MODIFICATION ---
        
        return image, label