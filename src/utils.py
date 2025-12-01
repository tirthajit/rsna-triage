import os
import torch
import numpy as np
import random
import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
