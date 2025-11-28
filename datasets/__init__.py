# multimodal-transformer-from-scratch/datasets/__init__.py

# This file makes the directory a Python package.

"""
This is the datasets/__init__.py module.

It exposes the main Dataset classes for easy access from other parts of the project.
Example:
    from datasets import CIFAR10Dataset, TinyShakespeareDataset
"""

from .cifar10 import CIFAR10Dataset
from .tinyshakespeare import TinyShakespeareDataset
from .Flickr8k import Flickr8kDataset

# You can also define a list of all public objects of the package.
__all__ = [
    "CIFAR10Dataset",
    "TinyShakespeareDataset",
    "Flickr8kDataset"
]