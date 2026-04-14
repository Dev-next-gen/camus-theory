# Pretrained Adapter Checkpoints

Trained TemporalAdapter weights are published as **GitHub Release assets** of this repository (not tracked in git).

| Checkpoint | Base model | Size | Hook layer | Train params |
|---|---|---|---|---|
| `graft_tl_ep2.pt` | TinyLlama-1.1B | 25 MB | 11 | 6.3M |
| `graft_qw14_ep2.pt` | Qwen2.5-14B | 63 MB | 23 | 15.8M |
| `graft_qwcoder32_ep2.pt` | Qwen2.5-Coder-32B | 63 MB | 31 | 15.8M |

## Download

Grab them from the latest release:
<https://github.com/Dev-next-gen/camus-theory/releases/latest>

Or with `gh`:
```bash
gh release download --pattern 'graft_*_ep2.pt' -R Dev-next-gen/camus-theory
```

## Loading a checkpoint

```python
import torch
from implementation.adapter.temporal_adapter import TemporalAdapter

ck = torch.load("graft_qw14_ep2.pt", map_location="cpu", weights_only=False)
adapter = TemporalAdapter(d_model=ck["d_model"], K=ck["K"], M=ck["M"])
adapter.load_state_dict(ck["adapter"])
```

See `implementation/inference/chat_qw14_local.py` for a complete multi-GPU inference loop with forward pre-hook injection.
