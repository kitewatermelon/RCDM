"""
diffusers UNet2DModel 팩토리.

컨디셔닝 방식 (diffusers 0.38 호환):
  ssl_feat → ssl_proj(Linear) → class_labels → class_embed_type="identity" → + time_emb → FiLM

diffusers 0.38의 UNet2DModel은 "projection" 타입을 지원하지 않으므로
"identity"를 사용하고, 외부에서 ssl_proj로 미리 time_embed_dim으로 투영한다.
원본 RCDM의 ssl_emb(Linear 1개)와 동일한 구조.
"""
from diffusers import UNet2DModel


def time_embed_dim(num_channels: int) -> int:
    """UNet2DModel의 time_embed_dim = block_out_channels[0] * 4."""
    return num_channels * 4


def create_unet(
    image_size: int = 128,
    num_channels: int = 256,
    num_res_blocks: int = 2,
    channel_mult: tuple = (1, 1, 2, 3, 4),
    attention_resolutions: tuple = (32, 16, 8),
    learn_sigma: bool = True,
    dropout: float = 0.0,
    use_scale_shift_norm: bool = True,
    attention_head_dim: int = 64,
) -> UNet2DModel:
    """
    Args:
        image_size: 입력 해상도 (정사각형)
        num_channels: 기본 채널 수
        channel_mult: 각 레벨별 채널 배수
        attention_resolutions: attention을 적용할 공간 해상도 목록
        learn_sigma: True이면 모델이 노이즈(3ch) + 분산(3ch) = 6ch 출력
        use_scale_shift_norm: FiLM 스타일 컨디셔닝 사용 여부
        attention_head_dim: attention head당 채널 수

    Note:
        SSL 컨디셔닝은 UNet 외부의 ssl_proj(nn.Linear)로 처리.
        RCDM 클래스에서 ssl_feat를 time_embed_dim으로 투영한 뒤
        class_labels로 전달하면 class_embed_type="identity"가 time_emb에 가산.
    """
    block_out_channels = tuple(num_channels * m for m in channel_mult)

    down_block_types = []
    up_block_types = []
    for i, _ in enumerate(channel_mult):
        res = image_size // (2 ** i)
        if res in attention_resolutions:
            down_block_types.append("AttnDownBlock2D")
        else:
            down_block_types.append("DownBlock2D")

    for bt in reversed(down_block_types):
        up_block_types.append(bt.replace("Down", "Up"))

    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=6 if learn_sigma else 3,
        block_out_channels=block_out_channels,
        layers_per_block=num_res_blocks,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        # class_labels를 time_emb에 그대로 가산 (RCDM 방식)
        class_embed_type="identity",
        # FiLM: GroupNorm(h) * (1+scale) + shift
        resnet_time_scale_shift="scale_shift" if use_scale_shift_norm else "default",
        dropout=dropout,
        norm_num_groups=32,
        attention_head_dim=attention_head_dim,
    )


def create_unet_128(**kwargs) -> UNet2DModel:
    """논문 기본 설정 (128×128) 단축 생성자."""
    return create_unet(
        image_size=128,
        num_channels=256,
        num_res_blocks=2,
        channel_mult=(1, 1, 2, 3, 4),
        attention_resolutions=(32, 16, 8),
        learn_sigma=True,
        dropout=0.0,
        use_scale_shift_norm=True,
        **kwargs,
    )
