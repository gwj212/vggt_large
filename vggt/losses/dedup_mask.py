# vggt/losses/dedup_mask.py
# ============================================================
# Per-pixel surface/fog mask, 用于 opacity 监督
#
# 两种实现:
#   compute_dedup_mask_image_domain  (★ 推荐, 速度快, 用 depth_conf 排序)
#   compute_dedup_mask_voxel         (慢, O(M²) KNN, 不适合 1M+ 点)
#
# 改动 v2.1 (相对 v2):
#   1. image_domain 的 depth_rel_tol 默认放宽 0.10 → 0.20,
#      让更多投影对齐到位置进入 conf 竞争 (减少误判遮挡)。
#   2. voxel 的 knn_k 默认 16 → 8, query_chunk 256 → 512,
#      速度提升一倍 (法线估计粗一点可接受)。
#   3. 文档警告: voxel 方案在 M > 5e5 时不适合训练循环.
# ============================================================

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_dedup_mask_image_domain(
    xyz_base,          # (B, S, H, W, 3) world points
    depth_conf,        # (B, S, H, W)    depth confidence
    extrinsics,        # (B, S, 3, 4)
    intrinsics,        # (B, S, 3, 3)
    H, W,
    z_near=0.05,
    depth_rel_tol=0.20,      # ★ v2.1: 0.10 → 0.20, 更宽容
    fallback_max_ratio=0.80,
    verbose=False,
):
    """
    图像域代表性分配:
      对每个 src 像素的世界点 P, 投到每个 obs 帧:
        in_fov & not_occluded & (obs.conf > src.conf) → src 让位

    Fallback:
      mask 比例 > fallback_max_ratio → 退化为全保留 (让 balance loss 接管)

    速度: ~50-100ms / batch, S=5, H=W=518
    """
    B, S, _, _, _ = xyz_base.shape
    device = xyz_base.device
    mask = torch.ones(B, S, H, W, device=device)

    world_maps = xyz_base.permute(0, 1, 4, 2, 3).reshape(B * S, 3, H, W)
    conf_maps = depth_conf.reshape(B * S, 1, H, W)

    for s in range(S):
        P_world = xyz_base[:, s].reshape(B, H * W, 3)
        conf_s = depth_conf[:, s].reshape(B, H * W)

        for o in range(S):
            if o == s:
                continue

            R = extrinsics[:, o, :3, :3]
            t = extrinsics[:, o, :3, 3]
            K = intrinsics[:, o]

            cam = P_world @ R.transpose(-1, -2) + t.unsqueeze(1)
            z_proj = cam[..., 2]

            fx = K[:, 0, 0].view(B, 1)
            fy = K[:, 1, 1].view(B, 1)
            cx = K[:, 0, 2].view(B, 1)
            cy = K[:, 1, 2].view(B, 1)

            u = fx * cam[..., 0] / z_proj.clamp(min=1e-6) + cx
            v = fy * cam[..., 1] / z_proj.clamp(min=1e-6) + cy

            in_fov = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z_proj > z_near)

            grid_u = 2.0 * u / max(W - 1, 1) - 1.0
            grid_v = 2.0 * v / max(H - 1, 1) - 1.0
            grid = torch.stack([grid_u, grid_v], dim=-1).view(B, H * W, 1, 2)

            obs_idx = torch.arange(B, device=device) * S + o
            obs_world_at_proj = F.grid_sample(
                world_maps[obs_idx], grid,
                mode='bilinear', align_corners=True,
            ).view(B, 3, H * W).transpose(-1, -2)

            obs_cam = obs_world_at_proj @ R.transpose(-1, -2) + t.unsqueeze(1)
            depth_obs_at_proj = obs_cam[..., 2]

            rel_diff = (z_proj - depth_obs_at_proj).abs() / depth_obs_at_proj.clamp(min=0.1)
            not_occluded = rel_diff < depth_rel_tol

            conf_obs_at_proj = F.grid_sample(
                conf_maps[obs_idx], grid,
                mode='bilinear', align_corners=True,
            ).view(B, H * W)

            higher_conf = conf_obs_at_proj > conf_s
            src_loses = in_fov & not_occluded & higher_conf

            mask[:, s].view(B, H * W)[src_loses] = 0.0

    # Fallback: mask 比例过高 → 退化为全保留
    for b in range(B):
        ratio = mask[b].mean()
        if ratio > fallback_max_ratio:
            if verbose:
                print(f"  [image-mask fallback] batch {b}: "
                      f"ratio={ratio*100:.1f}% > {fallback_max_ratio*100:.0f}%, "
                      f"退化为全保留")
            mask[b] = 1.0

    return mask


@torch.no_grad()
def compute_dedup_mask_voxel(
    xyz_base,
    depth_conf,
    voxel_size=0.02,
    normal_voxel_ratio=3.0,
    knn_k=8,                # ★ v2.1: 16 → 8
    query_chunk=512,        # ★ v2.1: 256 → 512
    fallback_min_ratio=0.15,
    verbose=False,
):
    """
    Voxel 方案: 法线方向各向异性体素 + 同体素 argmax(conf).

    ⚠️ 警告: KNN 复杂度 O(M²), M = S*H*W.
    M ≈ 1.34M (S=5, H=W=518) 时, 单次 mask 耗时 30-60 秒,
    训练循环里不可用. 仅作离线诊断 / 小规模实验用.
    """
    B, S, H, W, _ = xyz_base.shape
    device = xyz_base.device
    masks = []

    for b in range(B):
        pts = xyz_base[b].reshape(-1, 3)
        conf = depth_conf[b].reshape(-1)
        M = pts.shape[0]

        normals = torch.zeros_like(pts)
        for i in range(0, M, query_chunk):
            q = pts[i:i + query_chunk]
            d = torch.cdist(q, pts, p=2)
            _, idx = d.topk(knn_k, dim=-1, largest=False)
            nbrs = pts[idx]
            ctr = nbrs - nbrs.mean(dim=1, keepdim=True)
            cov = torch.einsum('ckm,ckn->cmn', ctr, ctr) / knn_k
            _, eigvec = torch.linalg.eigh(cov)
            normals[i:i + query_chunk] = eigvec[..., 0]
            del d, idx, nbrs, ctr, cov, eigvec

        up = torch.tensor([0., 1., 0.], device=device).expand_as(normals)
        u_axis = torch.cross(normals, up, dim=-1)
        bad = u_axis.norm(dim=-1) < 1e-4
        if bad.any():
            alt = torch.tensor([1., 0., 0.], device=device).expand_as(normals)
            u_axis = torch.where(
                bad.unsqueeze(-1),
                torch.cross(normals, alt, dim=-1),
                u_axis,
            )
        u_axis = F.normalize(u_axis, dim=-1)
        v_axis = F.normalize(torch.cross(normals, u_axis, dim=-1), dim=-1)

        local_u = (pts * u_axis).sum(-1) / voxel_size
        local_v = (pts * v_axis).sum(-1) / voxel_size
        local_n = (pts * normals).sum(-1) / (voxel_size * normal_voxel_ratio)

        keys = (
            local_u.floor().long() * 73856093 ^
            local_v.floor().long() * 19349663 ^
            local_n.floor().long() * 83492791
        )

        unique_keys, inv = torch.unique(keys, return_inverse=True)
        winner_conf = torch.full(
            (unique_keys.shape[0],), -float('inf'), device=device,
        )
        winner_conf.scatter_reduce_(0, inv, conf, reduce='amax', include_self=True)
        is_winner = (conf >= winner_conf[inv] - 1e-6).float()

        masks.append(is_winner.reshape(S, H, W))

    result = torch.stack(masks, dim=0)

    for b in range(B):
        ratio = result[b].mean()
        if ratio < fallback_min_ratio:
            if verbose:
                print(f"  [voxel-mask fallback] batch {b}: "
                      f"ratio={ratio*100:.1f}% < {fallback_min_ratio*100:.0f}%, "
                      f"退化为全保留")
            result[b] = 1.0

    return result


def compute_dedup_mask(method, xyz_base, depth_conf,
                       extrinsics=None, intrinsics=None,
                       H=None, W=None, verbose=False, **kwargs):
    """统一入口."""
    if method == 'image':
        return compute_dedup_mask_image_domain(
            xyz_base, depth_conf, extrinsics, intrinsics, H, W,
            verbose=verbose, **kwargs,
        )
    elif method == 'voxel':
        return compute_dedup_mask_voxel(
            xyz_base, depth_conf, verbose=verbose, **kwargs,
        )
    elif method == 'hybrid_and':
        m1 = compute_dedup_mask_image_domain(
            xyz_base, depth_conf, extrinsics, intrinsics, H, W)
        m2 = compute_dedup_mask_voxel(xyz_base, depth_conf)
        return m1 * m2
    elif method == 'hybrid_or':
        m1 = compute_dedup_mask_image_domain(
            xyz_base, depth_conf, extrinsics, intrinsics, H, W)
        m2 = compute_dedup_mask_voxel(xyz_base, depth_conf)
        return ((m1 + m2) > 0.5).float()
    elif method == 'none':
        B, S, H_, W_, _ = xyz_base.shape
        return torch.ones(B, S, H_, W_, device=xyz_base.device)
    else:
        raise ValueError(f"unknown dedup method: {method}")