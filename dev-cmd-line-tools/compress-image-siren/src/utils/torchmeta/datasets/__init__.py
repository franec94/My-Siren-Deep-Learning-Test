from src.utils.torchmeta.datasets.triplemnist import TripleMNIST
from src.utils.torchmeta.datasets.doublemnist import DoubleMNIST
from src.utils.torchmeta.datasets.cub import CUB
from src.utils.torchmeta.datasets.cifar100 import CIFARFS, FC100
from src.utils.torchmeta.datasets.miniimagenet import MiniImagenet
from src.utils.torchmeta.datasets.omniglot import Omniglot
from src.utils.torchmeta.datasets.tieredimagenet import TieredImagenet
from src.utils.torchmeta.datasets.tcga import TCGA

from src.utils.torchmeta.datasets import helpers

__all__ = [
    'TCGA',
    'Omniglot',
    'MiniImagenet',
    'TieredImagenet',
    'CIFARFS',
    'FC100',
    'CUB',
    'DoubleMNIST',
    'TripleMNIST',
    'helpers'
]
