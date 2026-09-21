# vggt/models/vggt.py  — v16 + depth_head 梯度 checkpoint (省显存, 让 5v 深度解冻不 OOM)
#
# 相对你给的 v16 唯一改动: depth_head 可训练(Stage-2)时, 对它的前向做梯度 checkpoint,
#   反向时重算 depth_head 前向以省下它的激活显存(代价≈多一次 depth_head 前向)。
#   → 让 "特征头 + 深度头同时解冻" 也能在 5 视角跑(否则 OOM, 之前被迫降到 3v)。
#
#   ★ 默认关 (VGGT_DEPTH_GRAD_CKPT != '1' 且 model._depth_grad_checkpoint 未设)
#     → 逐值等价你给的 v16, 不动任何已有实验。
#   ★ 仅在 self.training 且 depth_head 真可训练时启用 (eval / 冻结时零成本)。
#   ★ 开启: 启动加环境变量 VGGT_DEPTH_GRAD_CKPT=1 (无需改训练脚本)。
#
# 其余与 v16 完全一致。

import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as _torch_ckpt
from huggingface_hub import PyTorchModelHubMixin

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.heads.gaussian_head import GaussianHead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ============================================================
# Torch 版本的 depth -> world unproject (与 v16 一致)
# ============================================================

def unproject_depth_to_world_torch(
    depth_map: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    if depth_map.dim() == 5 and depth_map.shape[-1] == 1:
        depth_map = depth_map.squeeze(-1)
    assert depth_map.dim() == 4, f"unexpected depth shape: {depth_map.shape}"

    B, S, H, W = depth_map.shape
    device = depth_map.device
    dtype  = depth_map.dtype

    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij',
    )

    fu = intrinsics[..., 0, 0].view(B, S, 1, 1)
    fv = intrinsics[..., 1, 1].view(B, S, 1, 1)
    cu = intrinsics[..., 0, 2].view(B, S, 1, 1)
    cv = intrinsics[..., 1, 2].view(B, S, 1, 1)

    x_cam = (u_grid - cu) * depth_map / fu
    y_cam = (v_grid - cv) * depth_map / fv
    z_cam = depth_map
    cam_xyz = torch.stack([x_cam, y_cam, z_cam], dim=-1)

    R_w2c = extrinsics[..., :3, :3]
    t_w2c = extrinsics[..., :3,  3]
    R_c2w = R_w2c.transpose(-1, -2)

    cam_minus_t = cam_xyz - t_w2c.view(B, S, 1, 1, 3)

    world_xyz = torch.einsum('bsij,bshwj->bshwi', R_c2w, cam_minus_t)
    return world_xyz


# ============================================================
# VGGT model
# ============================================================

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        enable_gaussian=True,
        gaussian_xyz_offset_scale=0.1,
        gaussian_hidden_channels=256,
        gaussian_num_res_blocks=4,
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )

        self.camera_head = (
            CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        )
        self.point_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
            if enable_point
            else None
        )
        self.depth_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
            )
            if enable_depth
            else None
        )
        self.track_head = (
            TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
            if enable_track
            else None
        )

        if enable_gaussian:
            self.dpt_feature_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                features=256,
                feature_only=True,
            )
            self.gaussian_head = GaussianHead(
                in_channels=256,
                hidden_channels=gaussian_hidden_channels,
                num_res_blocks=gaussian_num_res_blocks,
                xyz_offset_scale=gaussian_xyz_offset_scale,
            )
        else:
            self.dpt_feature_head = None
            self.gaussian_head = None

        self._xyz_base_source_logged = False
        self._feat_grad_logged = False   # ★ v16: 首次 forward 打印一次特征头梯度通路状态
        self._depth_ckpt_logged = False   # ★ 首次 forward 打印一次 depth checkpoint 状态

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if query_points is not None and query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)

        B, S, _, H, W = images.shape

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        # Optional read-only feature export for downstream completion datasets.
        # It is disabled by default, so existing VGGT/3DGS training and inference
        # keep exactly the same outputs and memory lifetime.
        if getattr(self, "_return_completion_tokens", False):
            predictions["completion_tokens"] = aggregated_tokens_list[-1][
                :, :, patch_start_idx:
            ].detach()

        def tokens_for(head, detach=False):
            parameter = next(head.parameters())
            return [
                token.detach().to(device=parameter.device, dtype=parameter.dtype)
                if detach else token.to(device=parameter.device, dtype=parameter.dtype)
                for token in aggregated_tokens_list
            ]

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(tokens_for(self.camera_head))
                pose_enc_list = [pose.to(images.device) for pose in pose_enc_list]
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                # ★ depth_head 可训练(Stage-2)时, 做梯度 checkpoint 省激活显存 → 5v 不 OOM。
                #   默认关(env VGGT_DEPTH_GRAD_CKPT!=1 且未设 _depth_grad_checkpoint)= 逐值等价 v16。
                _depth_trainable = (self.training and
                                    any(p.requires_grad for p in self.depth_head.parameters()))
                _use_depth_ckpt = (_depth_trainable and
                                   (getattr(self, '_depth_grad_checkpoint', False)
                                    or os.environ.get('VGGT_DEPTH_GRAD_CKPT', '0') == '1'))

                if not self._depth_ckpt_logged:
                    print(f"[VGGT] depth_head grad-checkpoint = {_use_depth_ckpt} "
                          f"(depth_trainable={_depth_trainable})", flush=True)
                    self._depth_ckpt_logged = True

                if _use_depth_ckpt:
                    _psi = patch_start_idx
                    _depth_tokens = tokens_for(self.depth_head)

                    def _run_depth_head(_imgs, *_tokens):
                        return self.depth_head(list(_tokens), images=_imgs,
                                               patch_start_idx=_psi)

                    # use_reentrant=False: 支持非张量输出 + 正确保留 autocast 状态;
                    # depth_head 的参数虽不在入参里, 其梯度照常累积(标准用法)。
                    depth, depth_conf = _torch_ckpt.checkpoint(
                        _run_depth_head, images, *_depth_tokens,
                        use_reentrant=False,
                    )
                else:
                    depth, depth_conf = self.depth_head(
                        tokens_for(self.depth_head),
                        images=images,
                        patch_start_idx=patch_start_idx,
                    )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

                # ★ Stage-2: 冻结深度参照, 给锚定正则 ‖depth - depth_frozen‖ 用。
                #   仅当训练脚本注入了 self._depth_head_frozen 且在 training 时才算
                #   (eval / 未解冻深度时零成本)。frozen 输出在 no_grad 下, 天然 detach。
                _dhf = getattr(self, '_depth_head_frozen', None)
                if _dhf is not None and self.training:
                    with torch.no_grad():
                        _agg_det = tokens_for(_dhf, detach=True)
                        depth_frozen, _ = _dhf(
                            _agg_det, images=images.detach(),
                            patch_start_idx=patch_start_idx,
                        )
                    predictions["depth_frozen"] = depth_frozen

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    tokens_for(self.point_head),
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            # ============================================================
            # GaussianHead — ★ v16 关键改动 ★
            # ============================================================
            if self.gaussian_head is not None and self.dpt_feature_head is not None:
                # ★ v16: 给特征头的 tokens 先 detach → aggregator(backbone) 永远不收梯度,
                #   无论 backbone 是否冻结都安全。
                _agg_tokens_for_feat = tokens_for(
                    self.dpt_feature_head, detach=True
                )
                dpt_dtype = next(self.dpt_feature_head.parameters()).dtype
                dpt_feats = self.dpt_feature_head(
                    _agg_tokens_for_feat,
                    images=images.detach().to(dtype=dpt_dtype),
                    patch_start_idx=patch_start_idx,
                )
                gaussian_dtype = next(self.gaussian_head.parameters()).dtype
                dpt_feats = dpt_feats.to(dtype=gaussian_dtype)

                # ★★★ xyz_base 来源决策(支持环境变量切换, 同 v15.1)★★★
                _xyz_src = os.environ.get(
                    'VGGT_XYZ_BASE_SOURCE', 'depth_unproject'
                ).strip().lower()

                if not self._xyz_base_source_logged:
                    print(
                        f"[VGGT] xyz_base source = '{_xyz_src}' "
                        f"(VGGT_XYZ_BASE_SOURCE env var)",
                        flush=True,
                    )
                    self._xyz_base_source_logged = True

                if _xyz_src == 'world_points':
                    if "world_points" in predictions:
                        xyz_base = predictions["world_points"]
                    else:
                        xyz_base = torch.zeros(
                            B, S, H, W, 3,
                            device=images.device, dtype=images.dtype,
                        )
                else:
                    if (("depth" in predictions)
                            and ("pose_enc" in predictions)
                            and (predictions["depth"] is not None)):
                        extr, intr = pose_encoding_to_extri_intri(
                            predictions["pose_enc"],
                            image_size_hw=(H, W),
                        )
                        xyz_base = unproject_depth_to_world_torch(
                            predictions["depth"].float(),
                            extr.float(),
                            intr.float(),
                        )
                    elif "world_points" in predictions:
                        xyz_base = predictions["world_points"]
                    else:
                        xyz_base = torch.zeros(
                            B, S, H, W, 3,
                            device=images.device, dtype=images.dtype,
                        )

                # ─────────────────────────────────────────────────────────
                # ★ v16 改动核心:
                #   - dpt_feats 不再 .detach() → render loss 可训 dpt_feature_head
                #   - xyz_base 是否 detach 由 model._release_depth 控制(延迟解冻)
                #   - input_images 仍 .detach()
                # ─────────────────────────────────────────────────────────
                _release_depth = getattr(self, '_release_depth', False)
                _xyz_for_gh = xyz_base if _release_depth else xyz_base.detach()

                if not self._feat_grad_logged:
                    _trainable = any(p.requires_grad for p in self.dpt_feature_head.parameters())
                    _dep_train = (self.depth_head is not None
                                  and any(p.requires_grad for p in self.depth_head.parameters()))
                    print(
                        f"[VGGT] dpt_feats 梯度通路已开 (v16); dpt_feature_head "
                        f"{'可训练' if _trainable else '冻结'}; depth_head "
                        f"{'可训练(Stage-2)' if _dep_train else '冻结'}; "
                        f"xyz_base 解冻={_release_depth}",
                        flush=True,
                    )
                    self._feat_grad_logged = True

                gaussians = self.gaussian_head(
                    dpt_feats,                       # ★ v16: 不再 .detach()
                    _xyz_for_gh.to(dtype=gaussian_dtype),  # ★ Stage-2: _release_depth=True 时不 detach
                    input_images=images.detach().to(dtype=gaussian_dtype),
                )
                predictions["gaussians"] = gaussians

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images

        return predictions
