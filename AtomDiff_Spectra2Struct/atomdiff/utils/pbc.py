import torch
from torch import Tensor

from typing import Tuple

def periodic_radius_graph(x: Tensor, r: float, cell: Tensor, loop: bool = False) -> Tuple[Tensor, Tensor]:
    """Computes graph edges based on a cutoff radius for point cloud data with periodic boundaries.
    This implementation is bruteforce with O(N^2) complexity (per batch), but is very quick for small scale data.
    
    Args:
        x (Tensor): Point cloud coordinates with shape (N, D) where N is the number of nodes/particles,
            and D is the dimensionality of the coordinate space.
        r (float): Cutoff radius.
        cell (Tensor): Periodic cell dimensions with shape (D, D). Normally for 3D data the shape is (3, 3).
        loop (bool): Whether to include self-loops.

    Returns:
        edge_index (Tensor): Edge indices with shape (2, E) where E is the number of (directed) edges.
        edge_vec (Tensor): Edge vectors with shape (E, D).

    Notes:
        - Does not work for batched inputs.
        - Not tested with D != 3 dimensionality.
        - Not accurate for cells that are very oblique.
    """
    inv_cell = torch.linalg.pinv(cell)
    
    vec = x[None,:,:] - x[:,None,:]
    vec = vec - torch.round(vec @ inv_cell) @ cell
    dist = torch.linalg.norm(vec, dim=-1)
    dist += torch.eye(x.size(0), device=x.device) * (1. - float(loop)) * (r + 1.)
    edge_index = torch.nonzero(dist < r).T
    i, j = edge_index
    edge_vec = vec[i, j]
    return edge_index, edge_vec
