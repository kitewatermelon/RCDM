"""
RCDM 학습 스크립트 (HuggingFace Accelerate 기반).

사용 예:
    accelerate launch src/train.py \
        --data_dir /path/to/imagenet \
        --output_dir ./checkpoints \
        --ssl_model dino \
        --image_size 128 \
        --batch_size 8 \
        --num_epochs 500 \
        --lr 1e-4
"""
import argparse
import math
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .hybrid_loss import HybridLoss
from .model import create_unet, time_embed_dim
from .scheduler import create_ddpm_scheduler
from .ssl_models import get_ssl_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--ssl_model", type=str, default="dino",
                        choices=["dino", "simclr", "barlow", "vicreg", "supervised"])
    parser.add_argument("--use_head", action="store_true")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_channels", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=10, help="에폭 단위 체크포인트 저장 주기")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume_from", type=str, default=None, help="체크포인트 경로")
    return parser.parse_args()


def build_dataloader(data_dir: str, image_size: int, batch_size: int):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)


def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    accelerator.init_trackers("rcdm")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- SSL 인코더 (freeze) ---
    ssl_model = get_ssl_model(args.ssl_model, args.use_head)
    ssl_model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        ssl_dim = ssl_model(dummy).shape[1]

    # --- UNet ---
    unet = create_unet(
        image_size=args.image_size,
        num_channels=args.num_channels,
        learn_sigma=True,
    )
    # ssl_feat → time_embed_dim 투영 레이어 (학습 대상)
    t_dim = time_embed_dim(args.num_channels)
    ssl_proj = torch.nn.Linear(ssl_dim, t_dim)

    ema_model = EMAModel(
        list(unet.parameters()) + list(ssl_proj.parameters()),
        decay=args.ema_decay,
    )

    # --- Scheduler & Loss ---
    scheduler = create_ddpm_scheduler()
    loss_fn = HybridLoss(scheduler)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(ssl_proj.parameters()),
        lr=args.lr, betas=(0.9, 0.999),
    )
    train_loader = build_dataloader(args.data_dir, args.image_size, args.batch_size)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=len(train_loader) * args.num_epochs,
    )

    # --- Accelerate 준비 ---
    unet, ssl_proj, ssl_model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, ssl_proj, ssl_model, optimizer, train_loader, lr_scheduler
    )
    ema_model.to(accelerator.device)

    # --- Resume ---
    first_epoch = 0
    if args.resume_from is not None:
        accelerator.load_state(args.resume_from)
        # 체크포인트 디렉터리명에서 에폭 번호 파싱
        try:
            first_epoch = int(os.path.basename(args.resume_from).split("_")[-1]) + 1
        except ValueError:
            pass

    # --- 학습 루프 ---
    for epoch in range(first_epoch, args.num_epochs):
        unet.train()
        epoch_loss = 0.0

        for step, (images, _) in enumerate(train_loader):
            # images: [B, 3, H, W], [-1, 1]

            # SSL 특징 추출 → 투영 (ssl_proj는 학습 대상)
            with torch.no_grad():
                feat_raw = ssl_model(images)           # [B, ssl_dim]
            feat = ssl_proj(feat_raw)                  # [B, time_embed_dim]

            noise = torch.randn_like(images)
            B = images.shape[0]
            t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=images.device)
            x_t = scheduler.add_noise(images, noise, t)

            with accelerator.accumulate(unet):
                model_output = unet(x_t, t, class_labels=feat).sample
                loss, metrics = loss_fn(model_output, images, x_t, t, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(unet.parameters()) + list(ssl_proj.parameters()), 1.0
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                ema_model.step(unet.parameters())

            epoch_loss += metrics["loss"]

            if step % 100 == 0 and accelerator.is_main_process:
                accelerator.log(
                    {"loss": metrics["loss"], "mse": metrics["mse"], "vb": metrics["vb"],
                     "lr": lr_scheduler.get_last_lr()[0]},
                    step=epoch * len(train_loader) + step,
                )
                print(f"Epoch {epoch} | Step {step} | loss={metrics['loss']:.4f} "
                      f"mse={metrics['mse']:.4f} vb={metrics['vb']:.4f}")

        # 체크포인트 저장
        if (epoch + 1) % args.save_every == 0 and accelerator.is_main_process:
            save_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch:04d}")
            accelerator.save_state(save_dir)

            # EMA 가중치 별도 저장
            ema_unet = accelerator.unwrap_model(unet)
            ema_model.store(ema_unet.parameters())
            ema_model.copy_to(ema_unet.parameters())
            torch.save(ema_unet.state_dict(), os.path.join(save_dir, "unet_ema.pt"))
            ema_model.restore(ema_unet.parameters())

    accelerator.end_training()


if __name__ == "__main__":
    main()
