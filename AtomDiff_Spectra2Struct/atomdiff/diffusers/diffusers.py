import torch
from torch import Tensor

class VarianceExplodingDiffuser:
    def __init__(
        self,
        k: float = 1.,
        t_min: float = 1e-3,
        t_max: float = 0.999,
    ) -> None:
        self.t_min = t_min
        self.t_max = t_max

        # Case 1: sigma = kt
        self.alpha = lambda t: 1
        self.sigma = lambda t: k*t
        self.f     = lambda t: 0
        self.g2    = lambda t: 2*(k**2)*t
        self.g     = lambda t: self.g2(t)**0.5

    def forward_noise(self, x: Tensor, t: Tensor) -> Tensor:
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        eps = torch.randn_like(x)
        return alpha * x + sigma * eps, eps