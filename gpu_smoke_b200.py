"""GPU smoke test for the dosmatgen-cu128 environment.

Checks the three things that actually break on a new CUDA/GPU generation:

  1. the visible GPU, and whether torch was compiled with kernels for it
     (sm_100 for Blackwell-class cards such as the B200)
  2. a CUDA matmul + backward pass (the core cuBLAS / autograd path)
  3. a torch_scatter and a PyG message-passing op on CUDA — the compiled
     extensions are the real portability risk here, and a pure-torch test
     would not exercise them

Run it inside the environment built by build_cu128_env.sh, on a GPU node.
Checks 2 and 3 are meaningful on any CUDA GPU; check 1 only warns when the
card is not Blackwell-class.
"""
import torch

print("=" * 60)
print("torch", torch.__version__)
assert torch.cuda.is_available(), "no CUDA device visible - run this on a GPU node"
name = torch.cuda.get_device_name(0)
arch = torch.cuda.get_arch_list()
cap = torch.cuda.get_device_capability(0)
print("GPU:", name)
print("arch_list:", arch)
print("capability:", cap)

# --- 1: device generation vs compiled kernels ---
blackwell = cap[0] >= 10
if blackwell:
    assert any("sm_100" in a for a in arch), (
        f"Blackwell-class GPU ({name}, sm_{cap[0]}{cap[1]}) but sm_100 is not in "
        f"torch's compiled arch_list {arch} - this torch build cannot run here")
    print(f"[1] OK  Blackwell-class device ({name}) + sm_100 in compiled arch_list")
else:
    print(f"[1] SKIP  {name} is sm_{cap[0]}{cap[1]}, not Blackwell-class; "
          f"the sm_100 check does not apply. Checks 2-3 still validate the build.")

# --- 2: cuda matmul + backward ---
a = torch.randn(512, 512, device="cuda", requires_grad=True)
b = torch.randn(512, 512, device="cuda", requires_grad=True)
loss = (a @ b).sum()
loss.backward()
assert a.grad is not None and torch.isfinite(loss).item()
print(f"[2] OK  cuda matmul+backward  loss={loss.item():.3f}  grad_norm={a.grad.norm().item():.3f}")

# --- 3: torch_scatter / PyG message-pass on cuda ---
from torch_scatter import scatter_add
src = torch.ones(10, 4, device="cuda")
idx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], device="cuda")
out = scatter_add(src, idx, dim=0)
assert out.shape == (5, 4) and out.sum().item() == 40.0
print("[3a] OK  torch_scatter.scatter_add on cuda ->", tuple(out.shape))

# a real PyG message-pass layer (uses scatter under the hood) on cuda
from torch_geometric.nn import GCNConv
conv = GCNConv(4, 8).cuda()
x = torch.randn(5, 4, device="cuda")
edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], device="cuda")
y = conv(x, edge_index)
y.sum().backward()
assert y.shape == (5, 8) and y.is_cuda
print("[3b] OK  PyG GCNConv message-pass + backward on cuda ->", tuple(y.shape))

print("=" * 60)
print("ALL GPU SMOKE CHECKS PASSED")
