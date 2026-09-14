"""
AFOR: Adaptive Forgetting Optimizer.

Usage:
    from AFOR import afor

    optimizer = afor(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        beta2_min=0.99,
        weight_decay=1e-4,
    )
"""

import math

import torch
from torch.optim import Optimizer


class afor(Optimizer):
    """Tensor-wise adaptive forgetting optimizer."""

    def __init__(
        self,
        params,
        lr,
        betas=(0.9, 0.999),
        beta2_min=0.99,
        dir_weight=1.0,
        eps=1e-8,
        weight_decay=0,
        fast_ref=True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            beta2_min=beta2_min,
            dir_weight=dir_weight,
            eps=eps,
            weight_decay=weight_decay,
            fast_ref=fast_ref,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2_init = group["betas"]
            beta2_min = group["beta2_min"]
            dir_weight = group["dir_weight"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            fast_ref = group["fast_ref"]

            # beta2 is adapted within [beta2_min, beta2_init].
            beta2_max = beta2_init
            beta_fast = beta1
            beta_slow = beta2_init
            gate_warmup = 100

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("afor does not support sparse gradients")

                # Apply decoupled weight decay before the adaptive update.
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                state = self.state[p]

                # Initialize tensor-wise optimizer states.
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["noise_fast"] = torch.tensor(0.0, device=p.device)
                    state["noise_slow"] = torch.tensor(0.0, device=p.device)
                    state["dir_ema"] = torch.tensor(1.0, device=p.device)
                    state["beta2_cumprod"] = torch.tensor(1.0, device=p.device)
                    state["snr_ema"] = torch.tensor(0.0, device=p.device)
                    state["snr_var_ema"] = torch.tensor(1.0, device=p.device)

                    # These values are exposed for visualization and diagnostics.
                    state["snr_mag_t"] = torch.tensor(0.0, device=p.device)
                    state["snr_t"] = torch.tensor(0.0, device=p.device)
                    state["z_t"] = torch.tensor(0.0, device=p.device)
                    state["beta2_t"] = torch.tensor(1.0, device=p.device)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # The residual uses the previous first-moment estimate.
                noise_inst = (grad - exp_avg).abs().mean()
                state["noise_fast"].mul_(beta_fast).add_(
                    noise_inst, alpha=1 - beta_fast
                )
                state["noise_slow"].mul_(beta_slow).add_(
                    noise_inst, alpha=1 - beta_slow
                )

                # the previous momentum direction.
                dot = (grad * exp_avg).sum()
                ng, nm = grad.norm(), exp_avg.norm()
                if ng > eps and nm > eps:
                    cos = (dot / (ng * nm)).clamp(min=0.0)
                else:
                    cos = torch.tensor(1.0, device=p.device)
                state["dir_ema"].mul_(0.9).add_(cos, alpha=0.1)

                # momentum-to-noise signal ratio.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # combine relative momentum magnitude and direction
                # consistency into a non-negative tensor-wise signal score.
                if fast_ref:
                    noise_ref = torch.max(
                        state["noise_fast"], state["noise_slow"]
                    )
                else:
                    noise_ref = state["noise_slow"]

                mag_snr = exp_avg.abs().mean() / (noise_ref + eps)
                snr = mag_snr * (
                    1.0 + dir_weight * (state["dir_ema"] - 1.0)
                )
                snr = snr.clamp(min=0.0)

                # This avoids fixed, model-specific SNR thresholds.
                snr_val = snr.item()
                state["snr_ema"].mul_(beta_slow).add_(
                    snr_val, alpha=1 - beta_slow
                )
                delta_t = snr_val - state["snr_ema"]
                state["snr_var_ema"].mul_(beta_slow).addcmul_(
                    delta_t, delta_t, value=1 - beta_slow
                )

                var_clamped = max(state["snr_var_ema"].item(), 1e-8)
                z_score = delta_t / (math.sqrt(var_clamped) + eps)
                z_score = torch.clamp(z_score, -5.0, 5.0)

                beta2_raw = beta2_min + (
                    beta2_max - beta2_min
                ) * torch.sigmoid(z_score)

                # keep beta2 close to its initial value during the
                # warm-up period while the online statistics become reliable.
                gate = min(1.0, step / gate_warmup)
                beta2_t = gate * beta2_raw + (1.0 - gate) * beta2_init

                # Save observability values without changing the update rule.
                state["snr_mag_t"] = mag_snr
                state["snr_t"] = snr
                state["z_t"] = z_score
                state["beta2_t"] = beta2_t

                # update the second moment with time-varying beta2.
                state["beta2_cumprod"].mul_(beta2_t)
                beta2_value = beta2_t.item()
                exp_avg_sq.mul_(beta2_value).addcmul_(
                    grad, grad, value=1 - beta2_value
                )

                # corrections, using the cumulative product for time-varying
                # beta2, then update the parameters.
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = max(
                    1.0 - state["beta2_cumprod"].item(), 1e-12
                )

                denom = (
                    exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                ).add_(eps)
                p.addcdiv_(
                    exp_avg,
                    denom,
                    value=-(lr / bias_correction1),
                )

        return loss
