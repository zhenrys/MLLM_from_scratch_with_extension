# language_model/tokenizer.py

import json
import os
from typing import List, Dict

class CharacterTokenizer:
    """
    A character-level tokenizer that handles special tokens.
    """
    def __init__(self, corpus: str = None):
        self.special_tokens = {"pad": "<pad>", "sos": "<sos>", "eos": "<eos>", "unk": "<unk>"}
        self.chars: List[str] = []
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        
        self.pad_token_id = None
        self.sos_token_id = None
        self.eos_token_id = None
        self.unk_token_id = None
        
        self.pad_token = self.special_tokens["pad"]
        self.sos_token = self.special_tokens["sos"]
        self.eos_token = self.special_tokens["eos"]
        self.unk_token = self.special_tokens["unk"]

        self._initialize_vocab_with_special_tokens()
        if corpus:
            self.fit_on_text(corpus)

    def _initialize_vocab_with_special_tokens(self):
        for token_name, token_str in self.special_tokens.items():
            if token_str not in self.char_to_idx:
                idx = len(self.chars)
                self.chars.append(token_str)
                self.char_to_idx[token_str] = idx
                self.idx_to_char[idx] = token_str
        
        self.pad_token_id = self.char_to_idx[self.special_tokens["pad"]]
        self.sos_token_id = self.char_to_idx[self.special_tokens["sos"]]
        self.eos_token_id = self.char_to_idx[self.special_tokens["eos"]]
        self.unk_token_id = self.char_to_idx[self.special_tokens["unk"]]

    def fit_on_text(self, text: str):
        unique_chars = sorted(list(set(text)))
        for char in unique_chars:
            if char not in self.char_to_idx:
                idx = len(self.chars)
                self.chars.append(char)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char

    def get_vocab_size(self) -> int:
        return len(self.chars)
    
    def get_pad_token_id(self) -> int:
        return self.pad_token_id

    def get_sos_token_id(self) -> int:
        return self.sos_token_id
        
    def get_eos_token_id(self) -> int:
        return self.eos_token_id

    def encode(self, text_string: str) -> List[int]:
        """Encodes a string into a list of token IDs, without adding special tokens."""
        # --- START OF STUDENT MODIFICATION ---
        
        # TODO: 将输入的 `text_string` 转换为一个整数ID列表。
        #  - 遍历 `text_string` 中的每一个字符 `c`。
        #  - 在 `self.char_to_idx` 字典中查找字符 `c` 对应的索引。
        #  - 如果字符 `c` 不在字典中，则使用未知词元 `self.unk_token_id`。
        #  - 将所有索引收集到一个列表中并返回。
        #  - 提示: 列表推导式 (list comprehension) 是一个非常简洁的实现方式。
        return [self.char_to_idx.get(c, self.unk_token_id) for c in text_string ]

        # --- END OF STUDENT MODIFICATION ---

    def decode(self, int_list: List[int]) -> str:
        return ''.join([self.idx_to_char.get(i, '') for i in int_list if i not in [self.pad_token_id, self.sos_token_id, self.eos_token_id]])
    
    def __call__(self, text_string: str) -> List[int]:
        return self.encode(text_string)

    def save_vocab(self, vocab_path: str):
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        vocab_data = {
            'chars': self.chars, 'char_to_idx': self.char_to_idx,
            'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
            'special_tokens': self.special_tokens
        }
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {vocab_path}")

    def load_vocab(self, vocab_path: str):
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            
        self.chars = vocab_data['chars']
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        self.special_tokens = vocab_data.get('special_tokens', {"pad": "<pad>", "sos": "<sos>", "eos": "<eos>", "unk": "<unk>"})
        
        self.pad_token = self.special_tokens["pad"]
        self.sos_token = self.special_tokens["sos"]
        self.eos_token = self.special_tokens["eos"]
        self.unk_token = self.special_tokens["unk"]

        self._initialize_vocab_with_special_tokens()
        print(f"Vocabulary loaded from {vocab_path}")