"""
diffusers 스케줄러 설정 및 유틸리티.
DDPMScheduler (학습) / DDIMScheduler (빠른 샘플링) 제공.
"""
from diffusers import DDPMScheduler, DDIMScheduler


def create_ddpm_scheduler(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "linear",
    clip_sample: bool = True,
    prediction_type: str = "epsilon",
) -> DDPMScheduler:
    """
    학습용 DDPM 스케줄러.

    prediction_type="epsilon": 논문과 동일하게 노이즈 ε 예측.
    """
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        clip_sample=clip_sample,
        prediction_type=prediction_type,
    )


def create_ddim_scheduler(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "linear",
    clip_sample: bool = True,
    prediction_type: str = "epsilon",
) -> DDIMScheduler:
    """
    빠른 샘플링용 DDIM 스케줄러.
    set_timesteps(N)으로 N스텝으로 단축 가능 (예: 100스텝).
    """
    return DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        clip_sample=clip_sample,
        prediction_type=prediction_type,
    )
