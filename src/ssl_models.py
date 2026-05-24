"""
SSL 인코더 모델 정의.
get_ssl_models.py를 diffusers 기반 재구현에 맞게 이식.
"""
import torch
import torch.nn as nn
from torchvision import models as torchvision_models


class Wrapper(nn.Module):
    def __init__(self, model, head, use_head=False):
        super().__init__()
        self.model = model
        self.head = head
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.use_head = use_head

    def forward(self, x):
        x = self.model(x)
        if x.ndim > 2:
            x = self.pooling(x).view(x.size(0), -1)
        if self.use_head:
            x = self.head(x)
        return x


class SimCLRHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=2048, bottleneck_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        layers = [nn.Linear(in_dim, hidden_dim if nlayers > 1 else bottleneck_dim)]
        for i in range(nlayers - 2):
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        if nlayers > 1:
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x):
        return self.mlp(x)


def _projector(emb=8192):
    spec = f"2048-8192-8192-{emb}"
    f = list(map(int, spec.split("-")))
    layers = []
    for i in range(len(f) - 2):
        layers += [nn.Linear(f[i], f[i + 1]), nn.BatchNorm1d(f[i + 1]), nn.ReLU(True)]
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def get_ssl_model(model_name: str = "dino", use_head: bool = False) -> nn.Module:
    """
    SSL 인코더 반환. 반환된 모델은 eval 모드이며 gradient를 필요로 하지 않음.

    Args:
        model_name: "supervised" | "dino" | "simclr" | "barlow" | "vicreg"
        use_head: True이면 projection head 출력을 반환

    Returns:
        Wrapper(trunk, head, use_head) — forward(x) → feature vector
    """
    if model_name == "supervised":
        backbone = torchvision_models.resnet50(pretrained=True)
        backbone.fc = nn.Identity()
        return backbone.eval()

    elif model_name == "dino":
        backbone = torchvision_models.resnet50()
        backbone.fc = nn.Identity()
        head = DINOHead(2048, 60000, nlayers=2, use_bn=True)
        backbone.head = head
        ckpt = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth",
            map_location="cpu",
        )
        state = ckpt["teacher"]
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        backbone.load_state_dict(state, strict=True)
        return Wrapper(backbone, head, use_head).eval()

    elif model_name == "simclr":
        backbone = torchvision_models.resnet50()
        backbone.fc = nn.Identity()
        base = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch",
            map_location="cpu",
        )
        trunk_state = base["classy_state_dict"]["base_model"]["model"]["trunk"]
        trunk_state = {k.replace("_feature_blocks.", ""): v for k, v in trunk_state.items()}
        backbone.load_state_dict(trunk_state, strict=True)
        head = SimCLRHead()
        head_state = base["classy_state_dict"]["base_model"]["model"]["heads"]
        head_state = {k.replace("0.clf.0", "mlp.0").replace("1.clf.0", "mlp.2"): v for k, v in head_state.items()}
        head.load_state_dict(head_state, strict=True)
        return Wrapper(backbone, head, use_head).eval()

    elif model_name == "barlow":
        backbone = torchvision_models.resnet50()
        backbone.fc = nn.Identity()
        base = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/vissl/model_zoo/barlow_twins/barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch",
            map_location="cpu",
        )
        trunk_state = base["classy_state_dict"]["base_model"]["model"]["trunk"]
        trunk_state = {k.replace("_feature_blocks.", ""): v for k, v in trunk_state.items()}
        backbone.load_state_dict(trunk_state, strict=True)
        head = nn.Sequential(
            nn.Linear(2048, 8192, bias=False), nn.BatchNorm1d(8192), nn.ReLU(),
            nn.Linear(8192, 8192, bias=False), nn.BatchNorm1d(8192), nn.ReLU(),
            nn.Linear(8192, 8192, bias=False),
        )
        head_state = {k.replace("clf.", ""): v for k, v in base["classy_state_dict"]["base_model"]["model"]["heads"].items()}
        head.load_state_dict(head_state, strict=True)
        backbone.head = head
        return Wrapper(backbone, head, use_head).eval()

    elif model_name == "vicreg":
        backbone = torchvision_models.resnet50()
        backbone.fc = nn.Identity()
        projector = _projector(8192)
        backbone.projector = projector
        base = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/vicreg/resnet50_fullckpt.pth",
            map_location="cpu",
        )
        state = {k.replace("backbone.", "").replace("module.", ""): v for k, v in base["model"].items()}
        backbone.load_state_dict(state, strict=True)
        return Wrapper(backbone, projector, use_head).eval()

    else:
        raise ValueError(f"Unknown SSL model: {model_name!r}. Choose from: supervised, dino, simclr, barlow, vicreg")
