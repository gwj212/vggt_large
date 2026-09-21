# vggt/heads/gaussian_head.py  — v16 (v15 + ColorHead 可学性修复)
#
# 相对 v15 的唯一改动:
#   ★ ColorHead 最后一层不再 zero init, 改为极小随机(std=1e-3)。
#     原因: zero init 让 delta 初期严格=0, 起步梯度极弱, 实测 color_delta_norm 长期
#     卡在 ~0.01(几乎没学)。改极小随机后: 初期 color≈input(delta~1e-3 可忽略), 但
#     color head 一开始就有可观梯度, 能真正学起来。
#
#   ⚠ 注意(续训场景): 若从 ckpt resume, color_head 权重由 ckpt 覆盖, 本 init 改动不生效;
#     此时让颜色学起来的真正杠杆是训练脚本里的 COLOR_LR_MULTIPLIER(调大), 或在 load
#     之后手动重置 color_head(见训练脚本说明)。本改动主要利好"从头/重置训练 color"。
#
#   ★ 定位: 颜色只值 +3.65dB(color_refit 实测)且对重建/新视角(heldout)零贡献,
#     所以这是次要项, 主路径是 vggt.py 里解开几何。
#
# 其余与 v15 完全一致。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict


class ResidualBlock(nn.Module):
    """带 GroupNorm + GELU 的残差卷积块"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(32, channels // 4), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, channels // 4), channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class AttributeHead(nn.Module):
    """独立的属性预测头: 2层 1x1 conv (in→mid→out_dim)"""
    def __init__(self, in_channels: int, out_dim: int, mid_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_dim, 1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# ColorHead (v16: 极小随机 init)
# ============================================================

# ★ v16: color head 最后一层的 init std。1e-3 足够小, 初期 color≈input;
#   但比 0 有梯度, 能学起来。
COLOR_HEAD_INIT_STD = 1e-3


class ColorHead(nn.Module):
    """
    残差 color head, 输出 color = sigmoid(logit(input_rgb) + delta).
      - v16: 最后一层 weight 用极小随机(std=1e-3), bias=0 → 初期 delta≈0 但有梯度。
    """
    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid_channels, 3, 1, bias=True),
        )
        # ★ v16: 极小随机 init (取代 zero init), 让 color 一开始就有梯度
        nn.init.normal_(self.head[-1].weight, std=COLOR_HEAD_INIT_STD)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, input_rgb):
        rgb_safe = input_rgb.clamp(1e-4, 1 - 1e-4)
        rgb_logit = torch.log(rgb_safe / (1 - rgb_safe))

        delta = self.head(features)
        color = torch.sigmoid(rgb_logit + delta)
        return color, delta


class GaussianHead(nn.Module):
    """从 DPT feature_only 输出预测每像素 3D Gaussian 属性。(接口与 v15 一致)"""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        num_res_blocks: int = 4,
        head_mid_channels: int = 64,
        xyz_offset_scale: float = 0.1,
        frames_chunk_size: int = 1,
        use_checkpoint: bool = True,
        enable_color_head: bool = False,
    ):
        super().__init__()
        self.frames_chunk_size = frames_chunk_size
        self.use_checkpoint    = use_checkpoint
        self.enable_color_head = enable_color_head
        self.head_mid_channels = head_mid_channels
        self.hidden_channels   = hidden_channels

        self.xyz_proj = nn.Sequential(
            nn.Conv2d(3, 32, 1, bias=True),
            nn.GELU(),
        )
        fused_in = in_channels + 32

        self.stem = nn.Sequential(
            nn.Conv2d(fused_in, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden_channels // 4), hidden_channels),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        self.xyz_head     = AttributeHead(hidden_channels, 3, head_mid_channels)
        self.rot_head     = AttributeHead(hidden_channels, 4, head_mid_channels)
        self.scale_head   = AttributeHead(hidden_channels, 3, head_mid_channels)
        self.opacity_head = AttributeHead(hidden_channels, 1, head_mid_channels)

        if self.enable_color_head:
            self.color_head = ColorHead(hidden_channels, head_mid_channels)
        else:
            self.color_head = None

        self.xyz_offset_log_scale = nn.Parameter(
            torch.tensor(xyz_offset_scale).log()
        )

        self._init_weights()

    def _init_weights(self):
        """精心初始化，确保训练初期输出合理"""
        last_conv = self.xyz_head.net[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

        last_conv = self.rot_head.net[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)
        last_conv.bias.data[0] = 1.0

        last_conv = self.scale_head.net[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.constant_(last_conv.bias, -6.0)

        last_conv = self.opacity_head.net[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.constant_(last_conv.bias, 2.0)

        # ★ v16: ColorHead 改极小随机(取代 zero init)
        if self.color_head is not None:
            nn.init.normal_(self.color_head.head[-1].weight, std=COLOR_HEAD_INIT_STD)
            nn.init.zeros_(self.color_head.head[-1].bias)

    def enable_color_head_after_init(self, device=None, dtype=None):
        """在模型已创建后动态启用 color head (用于 from_pretrained 后再启用)。"""
        if self.color_head is not None:
            print("[GaussianHead] color_head 已存在, skip enable")
            return

        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        self.color_head = ColorHead(
            in_channels=self.hidden_channels,
            mid_channels=self.head_mid_channels,
        ).to(device=device, dtype=dtype)
        self.enable_color_head = True

        # ★ v16: 极小随机 init (取代 zero init)
        nn.init.normal_(self.color_head.head[-1].weight, std=COLOR_HEAD_INIT_STD)
        nn.init.zeros_(self.color_head.head[-1].bias)

        n_params = sum(p.numel() for p in self.color_head.parameters())
        print(f"[GaussianHead] ✓ ColorHead 已启用 ({n_params:,} params, "
              f"std={COLOR_HEAD_INIT_STD} init)")

    def reset_color_head(self):
        """★ v16 新增: 续训时若想让卡住的 color_head 重新开始学, 调它重置最后一层。"""
        if self.color_head is None:
            print("[GaussianHead] 无 color_head 可重置")
            return
        nn.init.normal_(self.color_head.head[-1].weight, std=COLOR_HEAD_INIT_STD)
        nn.init.zeros_(self.color_head.head[-1].bias)
        print(f"[GaussianHead] ✓ color_head 末层已重置 (std={COLOR_HEAD_INIT_STD})")

    def _trunk(self, dpt_feat_chunk, xyz_base_chunk_chw):
        xyz_proj = self.xyz_proj(xyz_base_chunk_chw)
        feat = torch.cat([dpt_feat_chunk, xyz_proj], dim=1)
        feat = self.stem(feat)
        for blk in self.res_blocks:
            if self.use_checkpoint and self.training:
                feat = checkpoint(blk, feat, use_reentrant=False)
            else:
                feat = blk(feat)
        return feat

    def forward(
        self,
        dpt_features: torch.Tensor,  # (B, S, C, H, W)
        xyz_base: torch.Tensor,      # (B, S, H, W, 3)
        input_images: torch.Tensor,  # (B, S, 3, H, W)
    ) -> Dict[str, torch.Tensor]:

        B, S, C, H, W = dpt_features.shape
        N = H * W

        feat = dpt_features.reshape(B * S, C, H, W)
        xyz_base_flat = (
            xyz_base.reshape(B * S, H, W, 3)
                    .permute(0, 3, 1, 2)
                    .contiguous()
        )
        input_images_flat = input_images.reshape(B * S, 3, H, W)

        if self.training and self.frames_chunk_size > 0:
            chunk = min(self.frames_chunk_size, B * S)
        else:
            chunk = B * S

        offset_scale = torch.exp(self.xyz_offset_log_scale)

        xyz_list, rot_list, scale_list, opa_list = [], [], [], []
        color_list = []
        delta_abs_sum = 0.0
        delta_count = 0

        for i in range(0, B * S, chunk):
            j = min(i + chunk, B * S)
            feat_c = feat[i:j]
            xyz_c  = xyz_base_flat[i:j]
            chunk_n = j - i

            h = self._trunk(feat_c, xyz_c)

            xyz_offset = self.xyz_head(h)
            rot_raw    = self.rot_head(h)
            scale_raw  = self.scale_head(h)
            opa_raw    = self.opacity_head(h)

            if self.enable_color_head and self.color_head is not None:
                rgb_c = input_images_flat[i:j]
                color_chw, delta_chw = self.color_head(h, rgb_c)
                color_chunk = (
                    color_chw.permute(0, 2, 3, 1).reshape(chunk_n, N, 3)
                )
                color_list.append(color_chunk)
                with torch.no_grad():
                    delta_abs_sum += delta_chw.abs().mean().item() * chunk_n
                    delta_count += chunk_n
                del delta_chw, color_chw

            xyz_chunk = xyz_c + xyz_offset * offset_scale
            xyz_chunk = (
                xyz_chunk.permute(0, 2, 3, 1).reshape(chunk_n, N, 3)
            )

            rot_chunk = (
                rot_raw.permute(0, 2, 3, 1).reshape(chunk_n, N, 4)
            )
            rot_chunk = F.normalize(rot_chunk, dim=-1)

            scale_chunk = (
                scale_raw.permute(0, 2, 3, 1).reshape(chunk_n, N, 3)
            )
            scale_chunk = torch.exp(scale_chunk.clamp(-10.0, 4.0))

            opa_chunk = (
                opa_raw.permute(0, 2, 3, 1).reshape(chunk_n, N, 1)
            )
            opa_chunk = torch.sigmoid(opa_chunk)

            xyz_list.append(xyz_chunk)
            rot_list.append(rot_chunk)
            scale_list.append(scale_chunk)
            opa_list.append(opa_chunk)

            del feat_c, xyz_c, h, xyz_offset, rot_raw, scale_raw, opa_raw

        xyz     = torch.cat(xyz_list, dim=0).reshape(B, S, N, 3)
        rot     = torch.cat(rot_list, dim=0).reshape(B, S, N, 4)
        scale   = torch.cat(scale_list, dim=0).reshape(B, S, N, 3)
        opacity = torch.cat(opa_list, dim=0).reshape(B, S, N, 1)

        if self.enable_color_head and self.color_head is not None:
            color = torch.cat(color_list, dim=0).reshape(B, S, N, 3)
            avg_delta_abs = delta_abs_sum / max(delta_count, 1)
        else:
            color = input_images.reshape(B * S, 3, H, W)
            color = color.permute(0, 2, 3, 1).reshape(B, S, N, 3).detach()
            avg_delta_abs = 0.0

        return {
            'xyz':      xyz,
            'rotation': rot,
            'scale':    scale,
            'opacity':  opacity,
            'color':    color,
            'color_delta_norm': avg_delta_abs,
        }