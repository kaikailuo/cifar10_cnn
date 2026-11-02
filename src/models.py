from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_layer(kind: str, channels: int, gn_groups: int = 8) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "ln":  # LayerNorm over (C, H, W) of each sample
        return nn.GroupNorm(1, channels)
    if kind == "in":  # InstanceNorm with affine parameters: one group per channel
        return nn.GroupNorm(channels, channels)
    if kind == "gn":
        return nn.GroupNorm(min(gn_groups, channels), channels)
    raise ValueError(f"unknown norm {kind!r}")


class BlurPool(nn.Module):
    """Low-pass filter then subsample: anti-aliased downsampling (Zhang, 2019)."""

    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        k = k[:, None] * k[None, :]
        self.register_buffer("kernel", (k / k.sum()).expand(channels, 1, 3, 3).clone())
        self.stride = stride
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x, self.kernel.to(x.dtype), stride=self.stride, groups=self.channels)


def downsample(pool: str, channels: int) -> nn.Module:
    if pool == "max":
        return nn.MaxPool2d(2)
    if pool == "blur":  # MaxBlurPool: dense max, then blurred subsampling
        return nn.Sequential(nn.MaxPool2d(2, stride=1), BlurPool(channels))
    raise ValueError(f"unknown pool {pool!r}")


class VGGNet(nn.Module):
    """VGG-style stages of [conv-norm-relu] x k + pooling, followed by a two-layer classifier."""

    def __init__(self, widths=(64, 128, 256), convs_per_stage=2, norm="none", gn_groups=8, pool="max",
                 conv_dropout=0.0, fc_hidden=256, fc_dropout=0.0, num_classes=10):
        super().__init__()
        layers: list[nn.Module] = []
        c = 3
        for w in widths:
            for _ in range(convs_per_stage):
                layers += [nn.Conv2d(c, w, 3, padding=1, bias=norm == "none"), norm_layer(norm, w, gn_groups),
                           nn.ReLU(inplace=True)]
                c = w
            layers.append(downsample(pool, w))
            if conv_dropout > 0:
                layers.append(nn.Dropout2d(conv_dropout))
        self.features = nn.Sequential(*layers)
        spatial = 32 // 2 ** len(widths)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * spatial * spatial, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, norm, gn_groups, residual):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.norm1 = norm_layer(norm, cout, gn_groups)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.norm2 = norm_layer(norm, cout, gn_groups)
        self.residual = residual
        self.shortcut = None
        if residual and (stride != 1 or cin != cout):
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), norm_layer(norm, cout, gn_groups))

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.residual:
            out = out + (x if self.shortcut is None else self.shortcut(x))
        return F.relu(out)


class ResNetCifar(nn.Module):
    """CIFAR ResNet of depth 6n + 2 (He et al., 2016); residual=False gives the equally deep plain network."""

    def __init__(self, depth=20, norm="bn", gn_groups=8, residual=True, widths=(16, 32, 64), num_classes=10):
        super().__init__()
        if (depth - 2) % 6:
            raise ValueError("resnet depth must be 6n + 2")
        n = (depth - 2) // 6
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), norm_layer(norm, widths[0], gn_groups),
                                  nn.ReLU(inplace=True))
        blocks, c = [], widths[0]
        for stage, w in enumerate(widths):
            for i in range(n):
                blocks.append(BasicBlock(c, w, 2 if stage > 0 and i == 0 else 1, norm, gn_groups, residual))
                c = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def _init_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def build_model(cfg: dict) -> nn.Module:
    if cfg["model"] == "vgg":
        model = VGGNet(cfg["widths"], cfg["convs_per_stage"], cfg["norm"], cfg["gn_groups"], cfg["pool"],
                       cfg["conv_dropout"], cfg["fc_hidden"], cfg["fc_dropout"])
    elif cfg["model"] == "resnet":
        model = ResNetCifar(cfg["depth"], cfg["norm"], cfg["gn_groups"], cfg["residual"])
    else:
        raise ValueError(f"unknown model {cfg['model']!r}")
    _init_weights(model)
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
