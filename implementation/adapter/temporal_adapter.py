"""
Temporal adapter — side-car brick injected into a frozen pretrained LM.

Input  : hidden states h (B, T, D), delta (B, T)
Output : h' = h + residual (same shape), MSE head for next-δ.

Trainable params scale with D and K·M. At D=2048, K=128, M=8 ≈ 80M.
"""
import math
import torch
import torch.nn as nn


class LogTimeTuning(nn.Module):
    def __init__(self, K=128, min_sec=0.05, max_sec=86400.0):
        super().__init__()
        mu = torch.linspace(math.log1p(min_sec), math.log1p(max_sec), K)
        self.mu = nn.Parameter(mu)
        self.log_sigma = nn.Parameter(torch.full((K,), math.log(0.5)))
        self.K = K

    def forward(self, delta):
        log_d = torch.log1p(delta.clamp(min=0)).unsqueeze(-1)
        sigma = self.log_sigma.exp()
        z = (log_d - self.mu) / (sigma + 1e-6)
        return torch.exp(-0.5 * z.pow(2))  # (B, T, K)


class LeakyCascade(nn.Module):
    def __init__(self, M=8, tau_min=1.0, tau_max=86400.0):
        super().__init__()
        tau = torch.logspace(math.log10(tau_min), math.log10(tau_max), M)
        self.register_buffer("tau", tau)
        self.M = M

    def forward(self, x, delta):
        # x: (B, T, D) ; delta: (B, T)
        B, T, D = x.shape
        dt = torch.zeros_like(delta)
        dt[:, 1:] = (delta[:, 1:] - delta[:, :-1]).clamp(min=0)
        alpha = torch.exp(-dt.unsqueeze(-1) / self.tau)  # (B, T, M)
        s = torch.zeros(B, self.M, D, device=x.device, dtype=x.dtype)
        out = []
        for t in range(T):
            a = alpha[:, t, :].unsqueeze(-1)
            s = a * s + (1 - a) * x[:, t, :].unsqueeze(1)
            out.append(s.reshape(B, self.M * D))
        return torch.stack(out, dim=1)  # (B, T, M*D)


class TemporalAdapter(nn.Module):
    """FiLM modulated by LogTimeTuning(δ) + LeakyCascade residual."""

    def __init__(self, d_model=2048, K=128, M=8, d_cond=256):
        super().__init__()
        self.tuning = LogTimeTuning(K=K)
        self.cond_proj = nn.Sequential(
            nn.Linear(K, d_cond), nn.GELU(), nn.Linear(d_cond, 2 * d_model))
        self.cascade = LeakyCascade(M=M)
        # Cascade output (M*D) is huge; bottleneck to d_cond then back.
        self.cascade_in = nn.Linear(d_model, d_cond, bias=False)
        self.cascade_out = nn.Linear(d_cond * M, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.head_next_delta = nn.Sequential(
            nn.Linear(d_model, d_cond), nn.GELU(), nn.Linear(d_cond, 1))
        self.d_model = d_model
        self.M = M

    def forward(self, h, delta):
        # FiLM from tuning
        tun = self.tuning(delta)                    # (B, T, K)
        g, b = self.cond_proj(tun).chunk(2, dim=-1)  # (B, T, D) each
        h_norm = self.ln(h)
        h_film = h_norm * (1 + g) + b
        # Cascade on bottleneck projection
        x_small = self.cascade_in(h_norm)           # (B, T, d_cond)
        casc = self.cascade(x_small, delta)         # (B, T, M*d_cond)
        h_casc = self.cascade_out(casc)             # (B, T, D)
        return h + h_film + h_casc

    def predict_next_delta(self, h):
        return self.head_next_delta(h).squeeze(-1)
