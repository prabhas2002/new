import torch
import random
import numpy as np
import torch.backends


class ArgStorage:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_gpu(gpu_id: int) -> torch.device:
    return torch.device(
        f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0\
        else 'mps' if torch.backends.mps.is_available() else 'cpu')
