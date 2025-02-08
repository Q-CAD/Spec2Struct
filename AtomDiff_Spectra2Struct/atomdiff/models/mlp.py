import torch
from torch import nn

from torch import Tensor
from typing import List, Tuple, Optional

class MLP(nn.Module):
    """
    multi-layer perceptron
    """
    def __init__(self, dims: List[int], act=None) -> None:
        super().__init__()
        self.dims = dims
        self.act = act

        num_layers = len(dims)

        layers = []
        for i in range(num_layers - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if (act is not None) and (i < num_layers - 2):
                layers += [act]
            
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dims={self.dims}, act={self.act})'