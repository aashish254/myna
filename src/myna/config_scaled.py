"""Free-fit scaled configs for T4 (≤16GB VRAM), targeting macro ≥0.60 from V1-B baseline of 0.4893."""
from dataclasses import dataclass

@dataclass
class MynaConfigScaledT4:
    """~36M param configuration optimized for T4 single instance (~2.3× V1-B's 15.4M)."""
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
    # Actual param count via MynaModel instantiation (approximation formula misses FFN weights)
    from myna.model import MynaModel
    model = MynaModel(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Actual (via MynaModel): {total_params:,} ({total_params/1_000_000:.2f}M)")
