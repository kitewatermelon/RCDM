"""
Improved DDPM Hybrid Loss (Nichol & Dhariwal 2021).

loss = MSE(ε, ε̂) + (T/1000) × VLB(v)

- ε̂ (앞 3채널): 노이즈 예측 → MSE
- v  (뒤 3채널): 분산 보간 파라미터 → KL divergence
  - v의 gradient는 분산 경로에만 흐름 (mean은 detach)
"""
import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler


def _normal_kl(mean1: torch.Tensor, logvar1: torch.Tensor,
               mean2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    """두 대각 Gaussian 간의 KL divergence."""
    return 0.5 * (
        -1.0
        + logvar2 - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2).pow(2) * torch.exp(-logvar2)
    )


def _approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """표준 정규분포 CDF 근사 (tanh 기반)."""
    return 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x.pow(3))))


def _discretized_gaussian_log_likelihood(
    x: torch.Tensor, means: torch.Tensor, log_scales: torch.Tensor
) -> torch.Tensor:
    """
    t=0에서 사용하는 이산 Gaussian 로그우도.
    x는 [-1, 1]로 정규화된 픽셀값 (255개 이산 레벨에 대응).
    """
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (x - means + 1.0 / 255.0)
    cdf_plus = _approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (x - means - 1.0 / 255.0)
    cdf_min = _approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = (cdf_plus - cdf_min).clamp(min=1e-12)

    return torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta)),
    )


class HybridLoss:
    """
    Hybrid MSE + VLB 손실 함수.

    Usage:
        scheduler = create_ddpm_scheduler()
        loss_fn = HybridLoss(scheduler)

        noise = torch.randn_like(x_start)
        x_t = scheduler.add_noise(x_start, noise, t)
        model_output = unet(x_t, t, class_labels=feat)
        loss, metrics = loss_fn(model_output, x_start, x_t, t, noise)
    """

    def __init__(self, scheduler: DDPMScheduler):
        self.T = scheduler.config.num_train_timesteps

        # 스케줄러로부터 posterior 관련 값 사전 계산 (GaussianDiffusion과 동일)
        betas = scheduler.betas.float()
        alphas_cumprod = scheduler.alphas_cumprod.float()
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # t=0에서 posterior_variance=0이 되므로 t=1 값으로 클리핑
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]])
        )
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.betas = betas
        self.posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * (1.0 - betas).sqrt() / (1.0 - alphas_cumprod)

    def _to(self, device: torch.device):
        """사전 계산 텐서를 지정 디바이스로 이동."""
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.betas = self.betas.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """배치 내 각 샘플의 timestep에 해당하는 값을 추출하여 [B,1,1,1] 형태로 반환."""
        return a[t].view(-1, 1, 1, 1)

    def _q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q(x_{t-1} | x_t, x_0)의 mean과 log_variance 반환.
        """
        mean = (
            self._extract(self.posterior_mean_coef1, t) * x_start
            + self._extract(self.posterior_mean_coef2, t) * x_t
        )
        log_var = self._extract(self.posterior_log_variance_clipped, t)
        return mean, log_var

    def _predict_x_start(
        self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor
    ) -> torch.Tensor:
        """ε̂로부터 x_0 예측."""
        abar = self._extract(self.alphas_cumprod, t)
        return (x_t - (1.0 - abar).sqrt() * eps_pred) / abar.sqrt()

    def _vb_loss(
        self,
        model_output: torch.Tensor,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        VLB 손실 계산.
        model_output의 mean 경로는 detach되어 gradient가 흐르지 않음.
        """
        eps_pred, var_values = model_output.chunk(2, dim=1)  # 각 [B,3,H,W]

        # true posterior: q(x_{t-1}|x_t, x_0)
        true_mean, true_log_var = self._q_posterior(x_start, x_t, t)

        # 예측 x_0 (mean 경로는 frozen)
        x0_pred = self._predict_x_start(x_t, t, eps_pred.detach()).clamp(-1.0, 1.0)
        p_mean, _ = self._q_posterior(x0_pred, x_t, t)

        # 예측 log variance: var_values ∈ (-1,1) → 선형 보간
        min_log = self._extract(self.posterior_log_variance_clipped, t)
        max_log = torch.log(self._extract(self.betas, t).clamp(min=1e-20))
        frac = (var_values + 1.0) / 2.0          # (0, 1) 범위로 변환
        p_log_var = min_log + frac * (max_log - min_log)

        # KL divergence (t > 0)
        kl = _normal_kl(true_mean, true_log_var, p_mean, p_log_var)
        kl = kl.mean(dim=[1, 2, 3]) / math.log(2.0)

        # 이산 NLL (t == 0)
        nll = -_discretized_gaussian_log_likelihood(x_start, p_mean, 0.5 * p_log_var)
        nll = nll.mean(dim=[1, 2, 3]) / math.log(2.0)

        return torch.where(t == 0, nll, kl)

    def __call__(
        self,
        model_output: torch.Tensor,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model_output: UNet 출력 [B, 6, H, W]
            x_start:      원본 이미지 [B, 3, H, W], [-1, 1] 정규화
            x_t:          노이즈 이미지 [B, 3, H, W]
            t:            타임스텝 [B]
            noise:        실제 추가된 노이즈 [B, 3, H, W]

        Returns:
            (loss, metrics_dict)
        """
        self._to(x_t.device)

        eps_pred, _ = model_output.chunk(2, dim=1)

        # Simple MSE loss on noise prediction
        mse = F.mse_loss(eps_pred, noise, reduction="none").mean(dim=[1, 2, 3])

        # VLB loss for variance learning
        vb = self._vb_loss(model_output, x_start, x_t, t)

        # Hybrid: MSE + (T/1000) * VLB  (Improved DDPM Eq.9)
        loss = mse + (self.T / 1000.0) * vb

        metrics = {
            "loss": loss.mean().item(),
            "mse": mse.mean().item(),
            "vb": vb.mean().item(),
        }
        return loss.mean(), metrics
