"""Interactive local inference of Qwen14B + temporal graft on 5x RX 7800 XT.

Usage:
  python chat_qw14_local.py --ckpt /mnt/DATA1/.../graft_qw14_ep2.pt \
                             --base /mnt/DATA1/MODELS/Qwen2.5-14B \
                             --alpha 0.2

REPL commands:
  /a <float>    set alpha (0..1, recommended 0.2)
  /d <float>    set delta seconds per token (e.g. 0.05, 3600, 86400)
  /r            reset conversation
  /q            quit
  anything else: sent as user message
"""
import argparse, os, sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "09_deploy_mi300x"))
from temporal_adapter import TemporalAdapter


class LocalGraft:
    def __init__(self, base_path, ckpt_path, alpha=0.2):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.hook_layer = ck["hook_layer"]
        self.d_model = ck["d_model"]
        self.K, self.M = ck["K"], ck["M"]
        self.alpha = alpha

        print(f"Loading base from {base_path} (bf16, device_map=auto)…")
        self.tok = AutoTokenizer.from_pretrained(base_path)
        self.base = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.base.eval()

        # Place adapter on same device as layer hook_layer+1 input
        target_layer = self.base.model.layers[self.hook_layer + 1]
        target_param = next(target_layer.parameters())
        self.device = target_param.device
        print(f"Hook target layer {self.hook_layer}, adapter device: {self.device}")

        self.adapter = TemporalAdapter(d_model=self.d_model, K=self.K, M=self.M)
        self.adapter.load_state_dict(ck["adapter"])
        self.adapter = self.adapter.to(self.device).to(torch.float32)
        self.adapter.eval()

        self._delta = None
        self._register_hook(target_layer)

    def _register_hook(self, target_layer):
        ad = self.adapter
        def pre_hook(module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if self._delta is None:
                return args, kwargs
            d = self._delta.to(h.device)
            with torch.no_grad():
                h32 = h.to(torch.float32)
                tun = ad.tuning(d)
                g, b = ad.cond_proj(tun).chunk(2, dim=-1)
                h_norm = ad.ln(h32)
                h_film = h_norm * (1 + g) + b
                x_small = ad.cascade_in(h_norm)
                casc = ad.cascade(x_small, d)
                h_casc = ad.cascade_out(casc)
                mod = self.alpha * (h_film - h_norm + h_casc)
            h_new = h + mod.to(h.dtype)
            if args:
                return (h_new,) + args[1:], kwargs
            else:
                kwargs["hidden_states"] = h_new
                return args, kwargs
        target_layer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    @torch.no_grad()
    def generate(self, messages, delta_rate=0.05, max_new=200, temp=0.7, top_p=0.9, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        text = ""
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        ids = self.tok.encode(text, return_tensors="pt").to(self.device)
        eos = self.tok.encode("<|im_end|>", add_special_tokens=False)[0]

        out_ids = []
        for _ in range(max_new):
            L = ids.size(1)
            self._delta = (torch.arange(L, device=self.device, dtype=torch.float32) * delta_rate).unsqueeze(0)
            logits = self.base(ids).logits
            self._delta = None
            nl = logits[0, -1, :].float() / max(temp, 1e-3)
            probs = F.softmax(nl, dim=-1)
            sp, si = probs.sort(descending=True)
            cum = sp.cumsum(0); mask = cum > top_p; mask[0] = False
            sp = sp.masked_fill(mask, 0); sp = sp / sp.sum()
            c = si[torch.multinomial(sp, 1)]
            tid = c.item()
            if tid == eos:
                break
            out_ids.append(tid)
            ids = torch.cat([ids, c.view(1, 1)], dim=1)
        return self.tok.decode(out_ids, skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/DATA1/WORKSPACE_IA/temporal_cognition/05_checkpoints/qw14/graft_qw14_ep2.pt")
    ap.add_argument("--base", default="/mnt/DATA1/MODELS/Qwen2.5-14B")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05, help="seconds per token")
    args = ap.parse_args()

    g = LocalGraft(args.base, args.ckpt, alpha=args.alpha)
    delta = args.delta
    messages = []
    print(f"\n=== Qwen14B + TemporalGraft — α={g.alpha} δ={delta}s/tok ===")
    print("Commands: /a <alpha>  /d <delta>  /r reset  /q quit\n")

    while True:
        try:
            u = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not u:
            continue
        if u.startswith("/q"):
            break
        if u.startswith("/r"):
            messages = []; print("(reset)"); continue
        if u.startswith("/a "):
            g.alpha = float(u[3:]); print(f"(alpha={g.alpha})"); continue
        if u.startswith("/d "):
            delta = float(u[3:]); print(f"(delta={delta}s/tok)"); continue
        messages.append({"role": "user", "content": u})
        resp = g.generate(messages, delta_rate=delta).split("<|im_end|>")[0].strip()
        print(f"bot> {resp}\n")
        messages.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    main()
