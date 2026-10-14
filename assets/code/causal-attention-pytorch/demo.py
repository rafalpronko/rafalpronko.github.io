import torch
import torch.nn.functional as F

from causal_attention import causal_attention

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

# One sequence, one head, 4 positions; D = 2 for Q/K and Dv = 3 for V.
q = torch.randn(1, 1, 4, 2)
k = torch.randn(1, 1, 4, 2)
v = torch.randn(1, 1, 4, 3)

out, weights = causal_attention(q, k, v)
print("weights[0, 0] (rows = query i, columns = key j):")
print(weights[0, 0])
print("row sums:", weights.sum(dim=-1)[0, 0])
print("output shape:", tuple(out.shape))

for dtype in (torch.float32, torch.float64):
    q2, k2, v2 = (x.to(dtype) for x in (q, k, v))
    mine, _ = causal_attention(q2, k2, v2)
    ref = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0, is_causal=True)
    print(f"max |mine - sdpa| {dtype}: {(mine - ref).abs().max().item():.3e}")

tokens = ["<bos>", "the", "cat", "sat", "down"]
inputs, targets = tokens[:-1], tokens[1:]
for i, (x, y) in enumerate(zip(inputs, targets)):
    print(f"position {i}: reads {inputs[: i + 1]} -> predicts {y!r}")

a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.5, is_causal=True)
b = F.scaled_dot_product_attention(q, k, v, dropout_p=0.5, is_causal=True)
print("dropout_p=0.5, two calls equal:", torch.equal(a, b))
