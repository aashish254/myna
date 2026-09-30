"""MynaConfig for 310–420M param model targeting ≥ 0.80 macro."""
from dataclasses import dataclass

@dataclass
class MynaConfigScaled:
    """1024-dim, 16-layer configuration derived from MynaConfig scaling plan."""
    vocab: int = 4096
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    d_k: int = 64
    d_v: int = 64
    d_ff: int = 2560
    d_ptr: int = 256
    
    def trunk(self):
        from myna.trunk import MynaTrunk
        return MynaTrunk(self.vocab, self.d_model, self.n_layers, self.n_heads, 
                         self.d_k, self.d_v, self.d_ff)


# Usage example (to be plumbed through train.py):
# config = MynaConfigScaled()
# model = MynaModelScaled(config)  # or update existing MynaModel to use this cfg


if __name__ == "__main__":
    cfg = MynaConfigScaled()
    print(f"v1-scaled config:")
    print(f"  vocab={cfg.vocab}, d_model={cfg.d_model}, n_layers={cfg.n_layers}")
    print(f"  n_heads={cfg.n_heads}, d_k={cfg.d_k}, d_v={cfg.d_v}")
    print(f"  d_ff={cfg.d_ff}, d_ptr={cfg.d_ptr}")
