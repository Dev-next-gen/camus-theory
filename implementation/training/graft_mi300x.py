"""
Graft TemporalAdapter onto frozen Qwen2.5-14B (bf16) on a single MI300X.
192 GB VRAM → no gradient checkpointing, big batch.

Usage:
    python graft_mi300x.py --model_path /path/to/Qwen2.5-14B --out_tag qw14

Args let you retarget other sizes (7B, 14B, 32B, 72B) without code change.
Hook layer defaults to half the depth if not specified.
"""
import argparse, os, sys, time, json, gzip
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_adapter import TemporalAdapter


class GraftedModel(nn.Module):
    def __init__(self, base, adapter, hook_layer):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.hook_layer = hook_layer
        self._delta_buf = None
        self._last_h = None
        for p in self.base.parameters():
            p.requires_grad_(False)

        def pre_hook(module, args, kwargs):
            if self._delta_buf is None:
                return None
            hs = args[0] if args else kwargs.get("hidden_states")
            dtype = hs.dtype
            hs_new = self.adapter(hs.float(), self._delta_buf).to(dtype)
            if args:
                args = (hs_new,) + args[1:]
                return args, kwargs
            else:
                kwargs["hidden_states"] = hs_new
                return args, kwargs

        self._h = self.base.model.layers[hook_layer + 1].register_forward_pre_hook(
            pre_hook, with_kwargs=True)

        def h_hook(module, inp, out):
            self._last_h = out if isinstance(out, torch.Tensor) else out[0]
        self._h2 = self.base.model.norm.register_forward_hook(h_hook)

    def forward(self, ids, delta):
        self._delta_buf = delta
        out = self.base(input_ids=ids, use_cache=False)
        self._delta_buf = None
        h = self._last_h
        pred_delta = self.adapter.predict_next_delta(h[:, :-1].float())
        return out.logits, h, pred_delta


def build_oasst_qwen(tokenizer, oasst_path, seq_len=256, max_conv=3000):
    from datetime import datetime
    from collections import defaultdict
    msgs = []
    with gzip.open(oasst_path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("deleted") or d.get("review_result") is False: continue
            if d.get("lang") != "en": continue
            msgs.append({"id": d["message_id"], "parent": d.get("parent_id"),
                         "ts": datetime.fromisoformat(str(d["created_date"])).timestamp(),
                         "text": d["text"], "role": d["role"]})
    children = defaultdict(list); roots = []
    for m in msgs:
        if m["parent"] is None: roots.append(m)
        else: children[m["parent"]].append(m)
    chains = []
    for root in roots:
        def longest(n):
            ks = children.get(n["id"], [])
            if not ks: return [n]
            return [n] + max((longest(k) for k in ks), key=len)
        c = longest(root)
        if len(c) >= 2: chains.append(c)
        if len(chains) >= max_conv: break
    RATE_U, RATE_A = 4.0, 20.0
    all_ids, all_ts = [], []
    for ch in chains:
        ids, ts = [], []
        for m in ch:
            role = "user" if m["role"] == "prompter" else "assistant"
            piece = f"<|im_start|>{role}\n{m['text'].strip()}<|im_end|>\n"
            tok_ids = tokenizer.encode(piece, add_special_tokens=False)
            rate = RATE_U if role == "user" else RATE_A
            dt = 1.0 / rate
            for i, tid in enumerate(tok_ids):
                ids.append(tid); ts.append(m["ts"] + i * dt)
        if len(ids) < seq_len: continue
        stride = seq_len // 4
        for i in range(0, len(ids) - seq_len + 1, stride):
            all_ids.append(ids[i:i+seq_len])
            t0 = ts[i]
            all_ts.append([t - t0 for t in ts[i:i+seq_len]])
    return (torch.tensor(all_ids, dtype=torch.long),
            torch.tensor(all_ts, dtype=torch.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--oasst_path", required=True)
    ap.add_argument("--out_dir", default="./out")
    ap.add_argument("--out_tag", default="qw14")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam_d", type=float, default=0.15)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--hook_layer", type=int, default=-1)
    ap.add_argument("--max_conv", type=int, default=3000)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    print(f"Loading base (bf16, frozen): {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)

    L = base.config.num_hidden_layers
    d = base.config.hidden_size
    hook = args.hook_layer if args.hook_layer >= 0 else (L // 2 - 1)
    print(f"  hidden={d}  layers={L}  hook_layer={hook}  vocab={base.config.vocab_size}")

    adapter = TemporalAdapter(d_model=d, K=args.K, M=args.M).to(device)
    model = GraftedModel(base, adapter, hook_layer=hook).to(device)

    n_trn = sum(p.numel() for p in adapter.parameters())
    print(f"  Trainable: {n_trn:,}")

    print("Tokenizing OASST (Qwen tokenizer)…")
    ids, deltas = build_oasst_qwen(tok, args.oasst_path, args.seq_len, args.max_conv)
    print(f"  {ids.shape[0]:,} sequences × {args.seq_len}")

    N = ids.size(0)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(42))
    split = int(0.95 * N)
    tr = perm[:split]
    train_ds = TensorDataset(ids[tr], deltas[tr])
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW([p for p in adapter.parameters()], lr=args.lr)

    print(f"Steps/epoch: {len(tl)}  |  eff_batch={args.batch_size*args.accum}")
    print(f"\n{'ep':>3} {'step':>5} {'LM':>7} {'L_δ':>7} {'t/s':>6}")

    t0 = time.monotonic(); gstep = 0
    for epoch in range(args.epochs):
        opt.zero_grad()
        micro = 0
        for ids_b, delta_b in tl:
            ids_b = ids_b.to(device, non_blocking=True)
            delta_b = delta_b.to(device, non_blocking=True)
            logits, h, pred = model(ids_b, delta_b)
            V = logits.size(-1)
            l_lm = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                                   ids_b[:, 1:].reshape(-1))
            dp = (delta_b[:, 1:] - delta_b[:, :-1]).clamp(min=0)
            l_d = F.mse_loss(pred, torch.log1p(dp))
            loss = l_lm + args.lam_d * l_d
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                opt.step(); opt.zero_grad(); gstep += 1
                if gstep % args.log_every == 0:
                    dt = time.monotonic() - t0
                    print(f"{epoch:>3} {gstep:>5} {l_lm.item():>7.3f} "
                          f"{l_d.item():>7.3f} {gstep/dt:>6.2f}")
        ck = os.path.join(args.out_dir, f"graft_{args.out_tag}_ep{epoch}.pt")
        torch.save({"adapter": adapter.state_dict(), "hook_layer": hook,
                    "K": args.K, "M": args.M, "epoch": epoch,
                    "base_path": args.model_path, "d_model": d, "L": L}, ck)
        print(f"  [ckpt] {ck}")

    report = {"trainable": n_trn, "epochs": args.epochs, "gsteps": gstep,
              "minutes": (time.monotonic()-t0)/60,
              "base_path": args.model_path, "d_model": d, "L": L, "hook_layer": hook}
    with open(os.path.join(args.out_dir, f"report_{args.out_tag}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
