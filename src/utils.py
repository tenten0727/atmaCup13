import random
import numpy as np
import os
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
