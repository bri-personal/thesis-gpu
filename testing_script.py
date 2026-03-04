import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


# following FlashAttention-3 paper
def generate_matrix(shape, seed=None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    # Base matrix from N(0, 1)
    base = np.random.normal(loc=0.0, scale=1.0, size=shape)
    # Bernoulli mask (0.001 probability of being 1)
    mask = np.random.binomial(n=1, p=0.001, size=shape)
    # Noise from N(0, 100)
    noise = np.random.normal(loc=0.0, scale=10.0, size=shape)
    # Final matrix: base + noise * mask
    return base + noise * mask


def scaled_dot_product_attention(Q_np: np.ndarray, K_np: np.ndarray, V_np: np.ndarray, causal: bool) -> np.ndarray:
    # ensure matching dimensions of 4D tensors
    assert (len(Q_np.shape), len(K_np.shape), len(V_np.shape)) == (4, 4, 4)
    b, h, seq_q, d = Q_np.shape
    bk, hk, seq_k, dk = K_np.shape
    bv, hv, seq_v, dv = V_np.shape
    assert b == 1 and b == bk and b == bv
    assert h == 1 and h == hk and h == hv
    assert d == dk, "Q and K head dim must be equal"
    assert d == dv, f"Q ({d}) and V ({dv}) head dim must be equal"
    assert seq_k == seq_v, "K and V must have equal seq len"

    # use CUDA on GPU
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    Q_torch = torch.from_numpy(Q_np).to(device)
    K_torch = torch.from_numpy(K_np).to(device)
    V_torch = torch.from_numpy(V_np).to(device)

    # for Turing arch, cannot use FlashAttention2
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        O_torch = F.scaled_dot_product_attention(Q_torch, K_torch, V_torch,
                                                 attn_mask=None,  # no masking
                                                 dropout_p=0.0,  # no dropout
                                                 is_causal=causal)

    return O_torch.cpu().numpy()


def main(seq_q, seq_kv, d, seed, causal):
    # Use FP16 for FA1 or MemEff Attention
    # Ensure 4D with correct axes to match MemEff implementation
    Q_np = generate_matrix((seq_q, d), seed=seed).astype(np.float16)[np.newaxis, np.newaxis, :, :]
    K_np = generate_matrix((seq_kv, d), seed=seed).astype(np.float16)[np.newaxis, np.newaxis, :, :]
    V_np = generate_matrix((seq_kv, d), seed=seed).astype(np.float16)[np.newaxis, np.newaxis, :, :]
    
    for _ in range(5):  # warm-up runs
        scaled_dot_product_attention(Q_np, K_np, V_np, causal)
    torch.cuda.synchronize()  # ensure all GPU work is done before timing
    
    O_np = scaled_dot_product_attention(Q_np, K_np, V_np, causal)
    torch.cuda.synchronize()
    print("Output shape:", O_np.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_q", type=int, default=256)
    parser.add_argument("--seq_kv", type=int, default=256)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--causal", action="store_true", default=False)
    args = parser.parse_args()

    main(args.seq_q, args.seq_kv, args.d, args.seed, args.causal)
