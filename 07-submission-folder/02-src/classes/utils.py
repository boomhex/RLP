from pathlib import Path
from typing import Tuple
import torch

def load_data(base: Path) -> torch.Tensor:
    x_train = load_tensor_file(base/"x_train.pt")
    x_test = load_tensor_file(base/"x_test.pt")
    y_train = load_tensor_file(base/"y_train.pt")
    y_test = load_tensor_file(base/"y_test.pt")
    return [x_train, x_test, y_train, y_test]

def load_tensor_file(dir: Path) -> Tuple:
    return torch.load(dir)