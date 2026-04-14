"""
All-in-one probe script for MI300X runs:
  1. Linear R² on log(1+δ_cum)
  2. Spearman distance preservation
  3. Multi-scale Fisher + accuracy
  4. SVD spectral analysis (time subspace identification)
  5. Counterfactual KL divergence + sample generations
"""
import argparse, os, sys, json, gzip
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_adapter import TemporalAdapter
from graft_mi300x import GraftedModel, build_oasst_qwen


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    base = AutoModelForCausalLM.from_pretrained(ck["base_path"], torch_dtype=torch.bfloat16).to(device)
    tok = AutoTokenizer.from_pretrained(ck["base_path"])
    ad = TemporalAdapter(d_model=ck["d_model"], K=ck["K"], M=ck["M"]).to(device)
    ad.load_state_dict(ck["adapter"])
    m = GraftedModel(base, ad, hook_layer=ck["hook_layer"]).to(device)
    m.eval()
    return m, tok, ck


@torch.no_grad()
def get_h(model, ids, delta):
    _, h, _ = model(ids, delta)
    return h.float()


def _ridge_solve(X, y, lam=1e-3):
    d = X.size(1)
    A = X.T @ X + lam * torch.eye(d, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ y)


def test_linear(model, ids, deltas, device, n=128):
    idx = torch.randperm(ids.size(0))[:n]
    ib = ids[idx].to(device); db = deltas[idx].to(device)
    h = get_h(model, ib, db)
    H = h.reshape(-1, h.size(-1)).to(device)
    y = torch.log1p(db.reshape(-1)).to(device)
    N = H.size(0); perm = torch.randperm(N, device=device); split = int(0.8 * N)
    tr, te = perm[:split], perm[split:]
    Htr = torch.cat([H[tr], torch.ones(split, 1, device=device)], dim=1)
    Hte = torch.cat([H[te], torch.ones(N - split, 1, device=device)], dim=1)
    w = _ridge_solve(Htr, y[tr].unsqueeze(-1))
    pred = (Hte @ w).squeeze(-1)
    ss_r = ((pred - y[te]) ** 2).sum()
    ss_t = ((y[te] - y[te].mean()) ** 2).sum() + 1e-9
    return (1 - ss_r / ss_t).item()


def test_distance(model, ids, deltas, device, n=128):
    idx = torch.randperm(ids.size(0))[:n]
    ib = ids[idx].to(device); db = deltas[idx].to(device)
    h = get_h(model, ib, db)
    dh_all, dt_all = [], []
    T = h.size(1)
    for b in range(min(32, h.size(0))):
        hb = h[b]; tb = torch.log1p(db[b])
        i = torch.randint(0, T, (200,)); j = torch.randint(0, T, (200,))
        m = i != j; i, j = i[m], j[m]
        dh_all.append((hb[i] - hb[j]).norm(dim=-1).cpu())
        dt_all.append((tb[i] - tb[j]).abs().cpu())
    dh = torch.cat(dh_all); dt = torch.cat(dt_all)
    rx = dh.argsort().argsort().float(); ry = dt.argsort().argsort().float()
    rx = (rx - rx.mean()) / (rx.std() + 1e-9)
    ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    return (rx * ry).mean().item()


def test_multiscale(model, device, T=128):
    scales = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    ids = torch.randint(100, 30000, (4, T), generator=torch.Generator().manual_seed(0)).to(device)
    Hs = []
    for dt in scales:
        delta = torch.full((4, T), dt, device=device); delta[:, 0] = 0.0
        delta_cum = torch.cumsum(delta, dim=1)
        h = get_h(model, ids, delta_cum)
        Hs.append(h.mean(dim=1))
    H = torch.cat(Hs, dim=0)
    y = torch.repeat_interleave(torch.arange(len(scales)), 4).to(device)
    mus = torch.stack([H[y == s].mean(0) for s in range(len(scales))])
    gm = H.mean(0)
    v_inter = ((mus - gm) ** 2).sum(dim=-1).mean().item()
    v_intra = torch.stack([((H[y == s] - mus[s]) ** 2).sum(dim=-1).mean()
                            for s in range(len(scales))]).mean().item()
    fisher = v_inter / (v_intra + 1e-9)
    H1 = torch.cat([H, torch.ones(H.size(0), 1, device=device)], dim=1)
    w = _ridge_solve(H1, y.unsqueeze(-1).float())
    pred = (H1 @ w).squeeze(-1).round().clamp(0, len(scales)-1).long()
    acc = (pred == y).float().mean().item()
    return fisher, acc


def test_svd(model, ids, deltas, device, n=128, q=64):
    idx = torch.randperm(ids.size(0))[:n]
    ib = ids[idx].to(device); db = deltas[idx].to(device)
    h = get_h(model, ib, db)
    H = h.reshape(-1, h.size(-1)).to(device)
    t = torch.log1p(db.reshape(-1)).to(device)
    mu = H.mean(0, keepdim=True); sd = H.std(0, keepdim=True) + 1e-6
    Hs = (H - mu) / sd
    U, S, Vh = torch.svd_lowrank(Hs, q=q, niter=4)
    scores = U * S
    cors = []
    for k in range(q):
        x = scores[:, k] - scores[:, k].mean()
        y = t - t.mean()
        r = (x * y).sum() / (x.norm() * y.norm() + 1e-9)
        cors.append(r.item())
    cors_abs = [abs(c) for c in cors]
    top5 = sorted(range(q), key=lambda k: -cors_abs[k])[:5]
    time_energy = sum(cors_abs[k] ** 2 for k in top5)
    pr = (S ** 2).sum() ** 2 / (S ** 4).sum()
    return {"top5_time_corr": [(k, cors[k], S[k].item()) for k in top5],
            "time_subspace_energy": time_energy,
            "participation_ratio": pr.item(),
            "top20_variance_pct": [(S[k] ** 2 / (S ** 2).sum() * 100).item() for k in range(20)]}


@torch.no_grad()
def counterfactual(model, tok, device, prompt, rates, n_new=25):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    outs, logs_by = {}, {}
    for label, rate in rates:
        gen = ids.clone(); logits_track = []
        for _ in range(n_new):
            T = gen.size(1)
            delta = (torch.arange(T, device=device, dtype=torch.float32) * rate).unsqueeze(0)
            logits, _, _ = model(gen, delta)
            nl = logits[0, -1, :].float()
            logits_track.append(nl.cpu())
            gen = torch.cat([gen, nl.argmax().unsqueeze(0).unsqueeze(0)], dim=1)
        outs[label] = tok.decode(gen[0], skip_special_tokens=True)[len(prompt):].strip()
        logs_by[label] = torch.stack(logits_track)
    ref = logs_by["1s"]
    kls = {}
    for label, logs in logs_by.items():
        pa = F.softmax(logs, -1); pb = F.softmax(ref, -1)
        kl = (pa * (torch.log(pa + 1e-12) - torch.log(pb + 1e-12))).sum(-1).mean().item()
        kls[label] = kl
    return outs, kls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--oasst_path", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--max_conv", type=int, default=500)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    print(f"Loading {args.ckpt}")
    model, tok, ck = load(args.ckpt, device)
    print("Tokenizing OASST…")
    ids, deltas = build_oasst_qwen(tok, args.oasst_path, args.seq_len, args.max_conv)
    print(f"  {ids.shape[0]} sequences")

    results = {"ckpt": args.ckpt, "base_path": ck["base_path"], "d_model": ck["d_model"], "L": ck["L"]}

    print("\n[1/5] Linear R²…")
    results["r2"] = test_linear(model, ids, deltas, device)
    print(f"  R² = {results['r2']:+.4f}")

    print("[2/5] Spearman…")
    results["rho"] = test_distance(model, ids, deltas, device)
    print(f"  ρ  = {results['rho']:+.4f}")

    print("[3/5] Multi-scale Fisher…")
    fisher, acc = test_multiscale(model, device)
    results["fisher"], results["acc_scales"] = fisher, acc
    print(f"  Fisher={fisher:.3f}  acc={acc:.1%}")

    print("[4/5] SVD…")
    results["svd"] = test_svd(model, ids, deltas, device)
    print(f"  time_subspace_energy={results['svd']['time_subspace_energy']:.3f}  PR={results['svd']['participation_ratio']:.2f}")
    for k, c, s in results["svd"]["top5_time_corr"]:
        print(f"    comp[{k:>2}] ρ={c:+.3f} σ={s:.1f}")

    print("[5/5] Counterfactual generation…")
    rates = [("0s", 0.0), ("0.05s", 0.05), ("1s", 1.0), ("1min", 60.0),
             ("1h", 3600.0), ("1day", 86400.0), ("1year", 31536000.0)]
    prompts = ["<|im_start|>user\nTell me a short story.<|im_end|>\n<|im_start|>assistant\n",
               "<|im_start|>user\nWhat happened next?<|im_end|>\n<|im_start|>assistant\n",
               "<|im_start|>user\nDescribe the scene.<|im_end|>\n<|im_start|>assistant\n"]
    results["counterfactual"] = []
    for p in prompts:
        outs, kls = counterfactual(model, tok, device, p, rates, n_new=25)
        results["counterfactual"].append({"prompt": p, "outputs": outs, "kl_vs_1s": kls})
        print(f"\n  prompt: {p[:60]}…")
        for label, _ in rates:
            print(f"    [{label:>6}] KL={kls[label]:>6.2f}  {outs[label][:100]}")

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    main()
