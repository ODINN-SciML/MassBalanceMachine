# lstm_mb_dan.py
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Gradient Reversal Layer
# -------------------------
class _GradReverseFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverseFn.apply(x, lambd)


# -------------------------
# Domain Discriminator
# -------------------------
class DomainDiscriminator(nn.Module):

    def __init__(
        self,
        in_dim: int,
        n_domains: int,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, D)
        return self.net(feat)  # (B, n_domains)


# -------------------------
# DAN Wrapper
# -------------------------
class LSTM_MB_DAN(nn.Module):
    """
    Domain-Adversarial (DAN/DANN-style) wrapper for LSTM_MB.

    - Keeps base model unchanged.
    - Uses base(..., return_features=True) to obtain fused sequence features z: (B,T,D)
    - Pools z -> feat: (B,D)
    - Applies GRL and a domain discriminator head.

    Forward returns: (y_month, y_w, y_a, domain_logits)
    """

    def __init__(
        self,
        base: nn.Module,  # your LSTM_MB instance
        n_domains: int,
        *,
        grl_lambda: float = 1.0,  # strength of gradient reversal
        dan_alpha: float = 0.1,  # weight on domain loss
        pool: str = "mean",  # "mean" or "last"
        disc_hidden: int = 128,
        disc_dropout: float = 0.1,
    ):
        super().__init__()
        self.base = base
        self.n_domains = int(n_domains)
        self.grl_lambda = float(grl_lambda)
        self.dan_alpha = float(dan_alpha)
        self.pool = str(pool)

        # infer fused_dim from base head input features
        if getattr(self.base, "two_heads", False):
            fused_dim = int(self.base.head_w.in_features)
        else:
            fused_dim = int(self.base.head.in_features)

        self.domain_disc = DomainDiscriminator(
            in_dim=fused_dim,
            n_domains=self.n_domains,
            hidden=int(disc_hidden),
            dropout=float(disc_dropout),
        )

    def pool_features(
        self, z: torch.Tensor, mv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        z: (B,T,D). Optionally use mv mask (B,T) to pool only valid months.
        Returns feat: (B,D)
        """
        if self.pool == "last":
            return z[:, -1, :]

        # default: mean pool
        if mv is None:
            return z.mean(dim=1)

        # masked mean pooling
        mv_f = mv.to(z.device).float()  # (B,T)
        denom = mv_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)
        feat = (z * mv_f.unsqueeze(-1)).sum(dim=1) / denom  # (B,D)
        return feat

    def forward(self, x_m, x_s, mv, mw, ma, *, domain_id=None, debug: bool = False):
        # get task outputs + fused features from base
        (y_month, y_w, y_a), z = self.base(
            x_m,
            x_s,
            mv,
            mw,
            ma,
            domain_id=domain_id,  # base may use adapter; safe either way
            debug=debug,
            return_features=True,
        )

        feat = self.pool_features(z, mv=mv)  # (B,D)
        feat = grad_reverse(feat, self.grl_lambda)  # (B,D) with reversed gradients
        dom_logits = self.domain_disc(feat)  # (B,n_domains)

        return y_month, y_w, y_a, dom_logits

    def dan_loss(self, outputs, batch, base_loss_fn) -> torch.Tensor:
        """
        outputs: (y_month, y_w, y_a, dom_logits)
        batch: must include 'domain_id' for the domain loss term
        base_loss_fn: your usual MB loss (e.g., LSTM_MB.custom_loss)
        """
        y_month, y_w, y_a, dom_logits = outputs
        loss_mb = base_loss_fn((y_month, y_w, y_a), batch)

        dom_y = batch.get("domain_id", None)
        if dom_y is None:
            # no domain labels -> behave like normal training
            return loss_mb

        if not torch.is_tensor(dom_y):
            dom_y = torch.tensor(dom_y, device=dom_logits.device, dtype=torch.long)
        dom_y = dom_y.to(dom_logits.device).long()

        loss_dom = F.cross_entropy(dom_logits, dom_y)
        return loss_mb + self.dan_alpha * loss_dom
