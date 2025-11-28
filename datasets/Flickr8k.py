# datasets/Flickr8k.py

"""
dataset download link：
https://www.kaggle.com/datasets/adityajn105/flickr8k

"""

import os
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
import torch
from language_model.tokenizer import CharacterTokenizer

class Flickr8kDataset(Dataset):
    """
    Flickr8k image captioning dataset.

    """
    def __init__(self, root: str, captions_file: str, transform=None, tokenizer: CharacterTokenizer=None):
        self.root = root
        self.transform = transform
        self.tokenizer = tokenizer

        assert self.tokenizer is not None, "A tokenizer must be provided."
        assert self.tokenizer.sos_token_id is not None, "Tokenizer must have a sos_token_id."
        assert self.tokenizer.eos_token_id is not None, "Tokenizer must have an eos_token_id."

        if os.path.isdir(os.path.join(self.root, "Images")):
            self.image_dir = os.path.join(self.root, "Images")
        elif os.path.isdir(os.path.join(self.root, "Flicker8k_Dataset")):
            self.image_dir = os.path.join(self.root, "Flicker8k_Dataset")
        else:
            raise RuntimeError("Image directory not found. Expected 'Images' or 'Flicker8k_Dataset'.")
            
        captions_path = os.path.join(self.root, captions_file)
        if not os.path.exists(captions_path):
            raise RuntimeError(f"Captions file not found at {captions_path}.")

        self.captions_map = self._load_captions(captions_path)
        
        self.samples = []
        for img_file, caption_list in self.captions_map.items():
            for caption in caption_list:
                self.samples.append((img_file, caption))

    def _load_captions(self, captions_path: str) -> dict:
        captions_map = defaultdict(list)
        with open(captions_path, 'r', encoding='utf-8') as f:
            next(f, None)
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(',', 1)
                if len(parts) == 2:
                    img_file, caption = parts
                    captions_map[img_file.strip()].append(caption.strip())
        return captions_map

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_file, caption_str = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_file)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}. Skipping to next sample.")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
        
        # --- START OF STUDENT MODIFICATION ---
        
        # TODO: 1. 将原始的 caption 字符串编码为 token ID 列表。
        #    - 使用 self.tokenizer.encode() 方法。
        caption_tokens = self.tokenizer.encode(caption_str)
        
        # TODO: 2. 构建模型所需的完整输入序列。
        #    - 序列必须以 <sos> token ID 开始。
        #    - 序列必须以 <eos> token ID 结束。
        #    - 提示: 使用 `self.tokenizer.get_sos_token_id()` 和 `self.tokenizer.get_eos_token_id()`。
        #    - 格式应为：[<sos_id>] + caption_tokens + [<eos_id>]
        #* ❌错误示范「int 拼 list」，model_input_sequence = [self.tokenizer.get_sos_token_id() + caption_tokens + self.tokenizer.get_eos_token_id()]
        model_input_sequence = (
            [self.tokenizer.get_sos_token_id()] +
            caption_tokens +
            [self.tokenizer.get_eos_token_id()]
        )
        # TODO: 3. 将最终的 ID 列表转换为 PyTorch 的 LongTensor。
        #    - 这是模型期望的输入格式。
        caption_output = torch.tensor(model_input_sequence, dtype=torch.long)
        
        # --- END OF STUDENT MODIFICATION ---

        return image, caption_output