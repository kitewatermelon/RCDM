"""
RCDM 메인 모듈 (diffusers 기반 재구현).

사용 예시:
    model = RCDM.from_pretrained("barlow", image_size=128)
    samples = model(image_batch, num_samples=4, num_inference_steps=100)
"""
import os
from typing import Optional

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel

from .model import create_unet_128, time_embed_dim
from .scheduler import create_ddim_scheduler, create_ddpm_scheduler
from .ssl_models import get_ssl_model

# UNet의 기본 num_channels (128×128 설정 기준)
_DEFAULT_NUM_CHANNELS = 256


class RCDM(nn.Module):
    """
    Representation Conditional Diffusion Model (diffusers 기반 재구현).

    컨디셔닝 흐름:
      ssl_feat [B, ssl_dim]
        → ssl_proj: Linear(ssl_dim, time_embed_dim)
        → class_labels [B, time_embed_dim]
        → UNet 내부에서 time_emb에 가산 (class_embed_type="identity")
        → 각 ResBlock에서 FiLM 적용
    """

    def __init__(
        self,
        unet: UNet2DModel,
        ssl_proj: nn.Linear,
        ssl_model: nn.Module,
        train_scheduler: DDPMScheduler,
        sample_scheduler: DDIMScheduler,
        image_size: int = 128,
    ):
        super().__init__()
        self.unet = unet
        self.ssl_proj = ssl_proj      # ssl_feat → time_embed_dim 투영
        self.ssl_model = ssl_model
        self.train_scheduler = train_scheduler
        self.sample_scheduler = sample_scheduler
        self.image_size = image_size

        for p in self.ssl_model.parameters():
            p.requires_grad_(False)

    @classmethod
    def build(
        cls,
        ssl_model_name: str = "dino",
        use_head: bool = False,
        image_size: int = 128,
        num_channels: int = _DEFAULT_NUM_CHANNELS,
        unet_checkpoint: Optional[str] = None,
    ) -> "RCDM":
        """
        SSL 모델과 UNet을 초기화하여 RCDM 인스턴스 생성.

        Args:
            ssl_model_name: SSL 인코더 종류
            use_head: True이면 projection head 출력을 조건으로 사용
            image_size: 이미지 해상도
            num_channels: UNet 기본 채널 수
            unet_checkpoint: 저장된 가중치 경로 (없으면 랜덤 초기화)
        """
        ssl_model = get_ssl_model(ssl_model_name, use_head)
        ssl_model.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            ssl_dim = ssl_model(dummy).shape[1]

        unet = create_unet_128()
        t_dim = time_embed_dim(num_channels)
        ssl_proj = nn.Linear(ssl_dim, t_dim)

        if unet_checkpoint is not None:
            state = torch.load(unet_checkpoint, map_location="cpu")
            unet.load_state_dict(state["unet"])
            ssl_proj.load_state_dict(state["ssl_proj"])

        train_scheduler = create_ddpm_scheduler()
        sample_scheduler = create_ddim_scheduler()

        return cls(unet, ssl_proj, ssl_model, train_scheduler, sample_scheduler, image_size)

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        num_samples: int = 1,
        num_inference_steps: int = 100,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        이미지 배치로부터 조건부 샘플 생성.

        Args:
            images: 조건 이미지 [B, 3, H, W], [-1, 1] 정규화 또는 [0, 1]
            num_samples: 각 조건 이미지당 생성할 샘플 수
            num_inference_steps: DDIM 스텝 수 (적을수록 빠름, 기본 100)
            generator: 재현성을 위한 RNG

        Returns:
            생성된 이미지 [B*num_samples, 3, H, W], [-1, 1] 범위
        """
        device = next(self.unet.parameters()).device
        images = images.to(device)
        B = images.shape[0]

        # SSL 특징 추출 → time_embed_dim으로 투영
        feat = self.ssl_model(images)               # [B, ssl_dim]
        feat = self.ssl_proj(feat)                  # [B, time_embed_dim]

        feat = feat.repeat_interleave(num_samples, dim=0)  # [B*num_samples, time_embed_dim]

        shape = (B * num_samples, 3, self.image_size, self.image_size)
        x = torch.randn(shape, device=device, generator=generator)

        self.sample_scheduler.set_timesteps(num_inference_steps, device=device)

        for t in self.sample_scheduler.timesteps:
            t_batch = t.expand(B * num_samples)
            # learn_sigma=True → 앞 3채널만 노이즈 예측으로 사용
            model_out = self.unet(x, t_batch, class_labels=feat).sample
            eps_pred = model_out[:, :3]

            x = self.sample_scheduler.step(eps_pred, t, x, generator=generator).prev_sample

        return x

    def get_feat(self, images: torch.Tensor) -> torch.Tensor:
        """SSL 특징 벡터 추출 (학습/분석용)."""
        with torch.no_grad():
            return self.ssl_model(images.to(next(self.ssl_model.parameters()).device))

    def interpolate(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        alphas: torch.Tensor,
        num_inference_steps: int = 100,
    ) -> torch.Tensor:
        """
        두 이미지의 SSL 표현 공간에서 선형 보간.

        Args:
            image_a: [1, 3, H, W]
            image_b: [1, 3, H, W]
            alphas:  [N] — 0.0(a)~1.0(b) 사이 보간 비율
            num_inference_steps: DDIM 스텝 수

        Returns:
            [N, 3, H, W]
        """
        device = next(self.unet.parameters()).device
        feat_a = self.ssl_proj(self.get_feat(image_a))  # [1, time_embed_dim]
        feat_b = self.ssl_proj(self.get_feat(image_b))  # [1, time_embed_dim]

        alphas = alphas.to(device).view(-1, 1)
        feat_interp = (1 - alphas) * feat_a + alphas * feat_b  # [N, time_embed_dim]

        shape = (len(alphas), 3, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)

        self.sample_scheduler.set_timesteps(num_inference_steps, device=device)
        for t in self.sample_scheduler.timesteps:
            t_batch = t.expand(len(alphas))
            model_out = self.unet(x, t_batch, class_labels=feat_interp).sample
            eps_pred = model_out[:, :3]
            x = self.sample_scheduler.step(eps_pred, t, x).prev_sample

        return x
