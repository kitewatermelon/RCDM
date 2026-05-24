"""
RCDM 샘플링 스크립트.

사용 예:
    # 기본 샘플링
    python -m src.sample \
        --data_dir /path/to/images \
        --ssl_model barlow \
        --unet_checkpoint ./checkpoints/checkpoint_epoch_0499/unet_ema.pt \
        --num_samples 4 \
        --num_inference_steps 100 \
        --output_dir ./samples

    # 보간
    python -m src.sample --mode interpolate \
        --image_a /path/a.jpg --image_b /path/b.jpg \
        --ssl_model barlow \
        --unet_checkpoint ./checkpoints/checkpoint_epoch_0499/unet_ema.pt
"""
import argparse
import os

import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

from .rcdm import RCDM


TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return TRANSFORM(img).unsqueeze(0)


def load_images_from_dir(data_dir: str, num_images: int) -> torch.Tensor:
    paths = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(data_dir)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:num_images]
    return torch.cat([load_image(p) for p in paths], dim=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="sample",
                        choices=["sample", "interpolate"])
    # 공통
    parser.add_argument("--ssl_model", type=str, default="barlow")
    parser.add_argument("--use_head", action="store_true")
    parser.add_argument("--unet_checkpoint", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # 샘플링 전용
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=4, help="조건으로 사용할 이미지 수")
    parser.add_argument("--num_samples", type=int, default=4, help="이미지당 생성 샘플 수")
    # 보간 전용
    parser.add_argument("--image_a", type=str, default=None)
    parser.add_argument("--image_b", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=8, help="보간 단계 수")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    model = RCDM.build(
        ssl_model_name=args.ssl_model,
        use_head=args.use_head,
        unet_checkpoint=args.unet_checkpoint,
    ).to(device)
    model.eval()

    if args.mode == "sample":
        assert args.data_dir is not None, "--data_dir 필요"
        images = load_images_from_dir(args.data_dir, args.num_images).to(device)
        print(f"조건 이미지 {len(images)}장으로 {args.num_samples}개씩 샘플링...")

        samples = model(images, num_samples=args.num_samples,
                        num_inference_steps=args.num_inference_steps)
        # [-1,1] → [0,1]
        samples = (samples + 1.0) / 2.0

        out_path = os.path.join(args.output_dir, "samples.png")
        vutils.save_image(samples, out_path, nrow=args.num_samples)
        print(f"저장: {out_path}")

    elif args.mode == "interpolate":
        assert args.image_a and args.image_b, "--image_a, --image_b 필요"
        img_a = load_image(args.image_a).to(device)
        img_b = load_image(args.image_b).to(device)
        alphas = torch.linspace(0.0, 1.0, args.num_steps)

        print(f"보간 {args.num_steps}단계...")
        interp = model.interpolate(img_a, img_b, alphas,
                                   num_inference_steps=args.num_inference_steps)
        interp = (interp + 1.0) / 2.0

        out_path = os.path.join(args.output_dir, "interpolation.png")
        vutils.save_image(interp, out_path, nrow=args.num_steps)
        print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
