# datasets/tinyshakespeare.py

"""
This is the datasets/tinyshakespeare.py module.

It provides a PyTorch Dataset class for the Tiny Shakespeare text corpus.
This version implements the classic block-based partitioning strategy, treating
the entire corpus as a single continuous stream of characters. This is a highly
efficient and standard approach for training autoregressive language models like GPT.
"""

import os
import torch
from torch.utils.data import Dataset
from .data_utils import download_file
# 假设此相对路径是正确的，如果您的项目结构不同，请相应调整
from language_model.tokenizer import CharacterTokenizer 

class TinyShakespeareDataset(Dataset):
    """
    Dataset for character-level language modeling on the Tiny Shakespeare corpus.
    
    This version treats the entire text corpus as one long sequence of characters.
    It generates training samples by sliding a window of `block_size` across this
    continuous sequence. This approach is efficient and excellent for learning
    continuous context.
    """
    def __init__(self, root: str = "data", block_size: int = 128, download: bool = True):
        self.url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data_dir = os.path.join(root, "tinyshakespeare")
        self.filepath = os.path.join(data_dir, "input.txt")
        self.vocab_path = os.path.join(data_dir, "vocab.json")
        self.block_size = block_size

        if download:
            download_file(self.url, self.filepath)
        
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset file not found at {self.filepath}. Please enable download=True.")

        with open(self.filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.tokenizer = CharacterTokenizer()
        if os.path.exists(self.vocab_path):
            print(f"Found existing vocabulary at {self.vocab_path}. Loading...")
            self.tokenizer.load_vocab(self.vocab_path)
        else:
            print(f"No vocabulary found. Building from corpus and saving to {self.vocab_path}...")
            self.tokenizer.fit_on_text(text)
            self.tokenizer.save_vocab(self.vocab_path)
            
        print("Encoding the entire corpus into a single sequence...")
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        print(f"Corpus successfully encoded. Total characters (tokens): {len(self.data)}")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, s: str) -> list[int]:
        return self.tokenizer.encode(s)

    def decode(self, l: list[int]) -> str:
        return self.tokenizer.decode(l)

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        [关键改动] 获取一个训练数据块。
        这个方法现在非常简单和高效。

        Args:
            idx (int): 数据块的起始索引。

        Returns:
            一个元组 (x, y)，其中：
            - x 是输入序列张量，形状为 (block_size,)
            - y 是目标序列张量，形状为 (block_size,)，是 x 向右移动一个位置的结果。
        """
        # --- START OF STUDENT MODIFICATION ---
        
        # TODO: 1. 从 self.data 中提取一个长度为 `self.block_size + 1` 的数据块。
        #    - 这个块将同时用于生成输入 `x` 和目标 `y`。
        #    - 起始位置由 `idx` 决定。
        chunk = self.data[idx : idx + self.block_size + 1]
        
        # TODO: 2. 创建输入序列 x。
        #    - `x` 应该是 `chunk` 的前 `block_size` 个字符。
        #    - 例如，如果 chunk 是 "hello", block_size 是 4, 那么 x 应该是 "hell"。
        #    - 提示: 使用 Python 的切片操作。
        x = chunk[:-1] #* 去掉最后一个字符
        
        # TODO: 3. 创建目标序列 y。
        #    - `y` 应该是 `chunk` 从第二个字符开始的 `block_size` 个字符。
        #    - `y` 的每个元素都是模型在看到 `x` 对应位置及之前所有输入后，需要预测的下一个字符。
        #    - 例如，如果 chunk 是 "hello", block_size 是 4, 那么 y 应该是 "ello"。
        #    - 提示: 使用 Python 的切片操作。
        y = chunk[1:] #* 去掉第一个字符
        
        # --- END OF STUDENT MODIFICATION ---
        
        return x, y