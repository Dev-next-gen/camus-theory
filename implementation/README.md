# CAMUS Theory — Reference Implementation

Minimal, self-contained reference code for the graft methodology described in Part II of the CAMUS Theory (DOI: [10.5281/zenodo.19557893](https://doi.org/10.5281/zenodo.19557893)).

## Layout

```
implementation/
├── adapter/
│   └── temporal_adapter.py   # TemporalAdapter: LogTimeTuning + LeakyCascade + FiLM
├── training/
│   ├── graft_mi300x.py       # Training entry point (single-file, MI300X-ready)
│   └── merge_shuffle.py      # Build the 60/40 Code-Feedback + OASST1 mix
├── inference/
│   └── chat_qw14_local.py    # Multi-GPU inference with runtime δ/α control
├── probes/
│   └── probes_mi300x.py      # Five evaluation probes (R², ρ, Fisher, multi-scale, SVD)
└── checkpoints/
    └── README.md             # Where to download trained adapters
```

## Requirements

- Python 3.10+
- PyTorch 2.9+ (CUDA or ROCm 7)
- `transformers >= 4.45`
- `datasets`, `numpy`, `scipy`

## Quickstart

### 1. Download a base model

```bash
# e.g. Qwen2.5-14B
huggingface-cli download Qwen/Qwen2.5-14B --local-dir ./models/Qwen2.5-14B
```

### 2. Train the adapter (MI300X or comparable)

```bash
python implementation/training/graft_mi300x.py \
  --model_path ./models/Qwen2.5-14B \
  --oasst_path ./data/oasst1.jsonl.gz \
  --out_dir ./out_qw14 \
  --out_tag qw14 \
  --hook_layer 23 \
  --seq_len 256 --batch_size 4 --accum 20 \
  --epochs 3 --lr 1e-4 --lam_d 0.15 \
  --K 128 --M 8
```

### 3. Run the evaluation probes

```bash
python implementation/probes/probes_mi300x.py \
  --ckpt ./out_qw14/graft_qw14_ep2.pt \
  --oasst_path ./data/oasst1.jsonl.gz \
  --out_json ./out_qw14/probes_qw14.json
```

### 4. Interactive inference

```bash
python implementation/inference/chat_qw14_local.py \
  --base ./models/Qwen2.5-14B \
  --ckpt ./out_qw14/graft_qw14_ep2.pt \
  --alpha 0.2 --delta 0.05
```

REPL commands: `/a <float>` set α · `/d <float>` set δ (s/tok) · `/r` reset · `/q` quit.

## Datasets

- **OASST1** — `OpenAssistant/oasst1` on Hugging Face ([arXiv:2304.07327](https://arxiv.org/abs/2304.07327))
- **Code-Feedback** — `m-a-p/Code-Feedback` ([arXiv:2402.14658](https://arxiv.org/abs/2402.14658))

Build the 60/40 mix for the Coder-32B run:

```bash
python implementation/training/merge_shuffle.py \
  --oasst ./data/oasst1.jsonl.gz \
  --code ./data/code_feedback_oasst.jsonl.gz \
  --out ./data/mix_code_oasst.jsonl.gz \
  --ratio_code 0.6
```

## License

Code: CC-BY-4.0, consistent with the Part II Zenodo record. See repository root `LICENSE`.
