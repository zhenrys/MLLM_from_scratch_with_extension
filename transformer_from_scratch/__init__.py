# transformer_from_scratch/__init__.py

"""
transformer_from_scratch package initializer.

This file makes the directory a Python package and exposes key classes
at the top level for easier imports.

Instead of writing:
from transformer_from_scratch.model import Transformer
from transformer_from_scratch.attention import MultiHeadAttention

You can write:
from transformer_from_scratch import Transformer, MultiHeadAttention
"""

# 从 attention.py 导入
from .attention import ScaledDotProductAttention, MultiHeadAttention

# 从 layers.py 导入
from .layers import PositionwiseFeedForward, PositionalEncoding, LayerNorm

# 从 blocks.py 导入
from .blocks import EncoderBlock, DecoderBlock

# 从 model.py 导入
from .model import TransformerEncoder, TransformerDecoder, Transformer

# 定义 __all__ 以便 'from transformer_from_scratch import *' 有明确的行为
__all__ = [
    'ScaledDotProductAttention',
    'MultiHeadAttention',
    'PositionwiseFeedForward',
    'PositionalEncoding',
    'LayerNorm',
    'EncoderBlock',
    'DecoderBlock',
    'TransformerEncoder',
    'TransformerDecoder',
    'Transformer',
]