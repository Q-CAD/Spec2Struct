import torch
from torch import Tensor, nn

class GaussianRandomFourierFeatures(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        input_dim: int = 1,
        sigma: float = 1.,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.sigma = sigma

        self.register_buffer('B', torch.randn(input_dim, embed_dim//2) * sigma)

    def forward(self, v: Tensor) -> Tensor:
        v_proj = 2 * torch.pi * v @ self.B
        return torch.cat([torch.cos(v_proj), torch.sin(v_proj)], dim=-1)