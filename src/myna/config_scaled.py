"""Free-fit scaled configs for T4 (≤16GB VRAM), targeting macro ≥0.60 from V1-B baseline of 0.4893."""
from dataclasses import dataclass

@dataclass
class MynaConfigScaledT4:
    """~80–100M param configuration optimized for T4 single instance."""
    vocab: int = 4096
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_k: int = 64
    d_v: int = 64
    d_ff: int = 1536
    d_ptr: int = 256
    
    def trunk(self):
        from myna.trunk import MynaTrunk
        return MynaTrunk(self.vocab, self.d_model, self.n_layers, self.n_heads, 
                         self.d_k, self.d_v, self.d_ff)


# Usage example (to be plumbed through train.py):
# config = MynaConfigScaledT4()
# model = MynaModelScaled(config)  # or update existing MynaModel to use this cfg


if __name__ == "__main__":
    cfg = MynaConfigScaledT4()
    print(f"v1-mid T4 config:")
    print(f"  vocab={cfg.vocab}, d_model={cfg.d_model}, n_layers={cfg.n_layers}")
    print(f"  n_heads={cfg.n_heads}, d_k={cfg.d_k}, d_v={cfg.d_v}")
    print(f"  d_ff={cfg.d_ff}, d_ptr={cfg.d_ptr}")
    # Approximate param count via formula: L × [4×d² + 4×d×H] per layer
    approx_trunk = cfg.n_layers * (4 * cfg.d_model**2 + 4 * cfg.d_model * cfg.n_heads)
    approx_probe = 2 * cfg.d_model * cfg.d_ptr
    total = approx_trunk + approx_probe
    print(f"  Estimated trunk params: ~{approx_trunk // 1_000_000}M")
    print(f"  Total (trunk+probe): ~{total // 1_000_000}M")
