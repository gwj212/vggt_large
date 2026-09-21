#!/usr/bin/env python3
"""
xyz_consistency.py - 跨帧 xyz 一致性 loss

核心思想:
    dedup_mask 告诉我们每个像素谁是 winner (mask=1) 谁是 loser (mask=0).
    Loser 像素的 3D 点应该和它在 winner 帧里的"对应表面"位置一致.
    我们把 loser 的 xyz 拉向 winner 表面上的对应点 (winner.detach()),
    单向拉力, 让跨帧 xyz 收敛到统一表面.

设计:
    1. Winner gradient detach (anchor)
    2. 对每个 loser 像素, 用其 xyz 投影到所有 winner 帧, 找投影像素的
       winner xyz 作为目标
    3. 距离用 L1 / scene_scale (尺度无关)
    4. mask 全 0 或全 1 的帧跳过

接入:
    from vggt.losses.xyz_consistency import compute_xyz_consistency_loss
    l_xyz_cons = compute_xyz_consistency_loss(
        pred_xyz=predictions['gaussians']['xyz'],  # (B, S, H*W, 3)
        dedup_mask=dedup_mask,                      # (B, S, H, W)
        extrinsics=extr,                            # (B, S, 4, 4) or (B, S, 3, 4)
        intrinsics=intr,                            # (B, S, 3, 3)
        H=H, W=W,
    )
"""

import torch
import torch.nn.functional as F


_EPS = 1e-6


@torch.no_grad()
def _scene_scale(xyz):
    """Estimate scene scale for distance normalization.

    Args:
        xyz: (..., 3)
    Returns:
        scalar tensor, average L2 norm
    """
    return xyz.reshape(-1, 3).norm(dim=-1).mean().clamp_min(_EPS)


def _to_homogeneous_4x4(extr):
    """Ensure extrinsics are 4x4.

    Args:
        extr: (B, S, 3, 4) or (B, S, 4, 4)
    Returns:
        (B, S, 4, 4)
    """
    if extr.shape[-2:] == (4, 4):
        return extr
    assert extr.shape[-2:] == (3, 4), \
        f"extrinsics shape expected (...,3,4) or (...,4,4), got {extr.shape}"
    B, S = extr.shape[:2]
    bottom = torch.zeros(B, S, 1, 4, device=extr.device, dtype=extr.dtype)
    bottom[..., 0, 3] = 1.0
    return torch.cat([extr, bottom], dim=-2)


def _project_world_to_pixel(xyz_world, extr_4x4, intr_3x3):
    """Project world-space points into each frame's pixel coords.

    Args:
        xyz_world: (B, N, 3)  world-space points
        extr_4x4:  (B, S, 4, 4) world-to-camera
        intr_3x3:  (B, S, 3, 3) camera intrinsics

    Returns:
        pixel_uv:  (B, S, N, 2)  pixel coordinates (u=x, v=y)
        depth:     (B, S, N)     depth in each camera (z, positive in front)
        valid:     (B, S, N)     bool, depth > 0
    """
    B, N, _ = xyz_world.shape
    S = extr_4x4.shape[1]

    # World → camera. Need homogeneous coords.
    xyz_h = torch.cat([xyz_world, torch.ones(B, N, 1, device=xyz_world.device,
                                             dtype=xyz_world.dtype)], dim=-1)  # (B, N, 4)
    # (B, S, 4, 4) @ (B, S, 4, N) = (B, S, 4, N), broadcast xyz over S
    xyz_h_exp = xyz_h.unsqueeze(1).expand(B, S, N, 4)                          # (B, S, N, 4)
    xyz_cam = torch.einsum('bsij,bsnj->bsni', extr_4x4, xyz_h_exp)             # (B, S, N, 4)
    xyz_cam = xyz_cam[..., :3]                                                  # (B, S, N, 3)

    depth = xyz_cam[..., 2]                                                     # (B, S, N)
    valid = depth > _EPS

    # Project to image plane: x/z, y/z, then K @ [x/z, y/z, 1]
    xy = xyz_cam[..., :2] / depth.unsqueeze(-1).clamp_min(_EPS)                # (B, S, N, 2)
    ones = torch.ones_like(xy[..., :1])
    xy1 = torch.cat([xy, ones], dim=-1)                                         # (B, S, N, 3)
    pixel = torch.einsum('bsij,bsnj->bsni', intr_3x3, xy1)                     # (B, S, N, 3)
    pixel_uv = pixel[..., :2]                                                   # (B, S, N, 2)

    return pixel_uv, depth, valid


def _grid_sample_xyz(xyz_per_frame, pixel_uv, H, W):
    """Sample xyz_per_frame at pixel_uv using bilinear grid_sample.

    Args:
        xyz_per_frame: (B, S, H, W, 3)  per-frame xyz map
        pixel_uv:      (B, S, N, 2)     pixel coords to sample
        H, W: int
    Returns:
        sampled_xyz:   (B, S, N, 3)
        valid_uv:      (B, S, N) bool, in-bounds
    """
    B, S, _, _, _ = xyz_per_frame.shape
    N = pixel_uv.shape[2]

    # Normalize pixel coords to [-1, 1] for grid_sample
    # pixel u in [0, W-1] → norm_x in [-1, 1]
    # pixel v in [0, H-1] → norm_y in [-1, 1]
    u = pixel_uv[..., 0]                                                        # (B, S, N)
    v = pixel_uv[..., 1]
    valid_uv = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)

    norm_x = 2.0 * u / max(W - 1, 1) - 1.0
    norm_y = 2.0 * v / max(H - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)                                # (B, S, N, 2)
    grid = grid.unsqueeze(2)                                                    # (B, S, 1, N, 2)

    # grid_sample expects input as (N, C, H, W), grid (N, H_out, W_out, 2)
    # We process per-(B,S) by merging:
    BS = B * S
    inp = xyz_per_frame.reshape(BS, H, W, 3).permute(0, 3, 1, 2).contiguous()   # (BS, 3, H, W)
    grd = grid.reshape(BS, 1, N, 2)                                             # (BS, 1, N, 2)

    sampled = F.grid_sample(inp, grd, mode='bilinear',
                            padding_mode='zeros', align_corners=True)            # (BS, 3, 1, N)
    sampled = sampled.squeeze(2).permute(0, 2, 1).contiguous()                   # (BS, N, 3)
    sampled = sampled.reshape(B, S, N, 3)

    return sampled, valid_uv


def compute_xyz_consistency_loss(
    pred_xyz,
    dedup_mask,
    extrinsics,
    intrinsics,
    H,
    W,
    min_winner_ratio_per_frame=0.01,
    min_loser_ratio_per_frame=0.01,
    max_loser_per_frame=20000,
):
    """Cross-frame xyz consistency loss.

    For each loser pixel in frame i:
      1. Project its predicted 3D point into all OTHER frames (j != i).
      2. In each frame j, sample the predicted xyz at the projected pixel.
      3. If the destination pixel is a winner (mask_j=1), use that xyz as target.
      4. Loss = ||loser_xyz - winner_xyz.detach()||_1 / scene_scale.

    Winner xyzs are detached (anchor). Loser xyzs receive gradient.

    Args:
        pred_xyz:    (B, S, H*W, 3)   per-pixel predicted xyz
        dedup_mask:  (B, S, H, W)     0/1 winner mask
        extrinsics:  (B, S, 4, 4) or (B, S, 3, 4)
        intrinsics:  (B, S, 3, 3)
        H, W: int

    Returns:
        loss (scalar tensor). 0 if no valid loser-winner pairs found.
        info dict with diagnostics.
    """
    B, S, HW, _ = pred_xyz.shape
    assert HW == H * W, f"pred_xyz HW={HW} but H*W={H*W}"
    assert dedup_mask.shape == (B, S, H, W), \
        f"dedup_mask shape mismatch: {dedup_mask.shape} vs (B,S,H,W)=({B},{S},{H},{W})"

    device = pred_xyz.device
    dtype  = pred_xyz.dtype

    extr_4x4 = _to_homogeneous_4x4(extrinsics.to(device).to(dtype))             # (B, S, 4, 4)
    intr_3x3 = intrinsics.to(device).to(dtype)                                   # (B, S, 3, 3)

    # Reshape pred_xyz to per-frame xyz map
    xyz_map = pred_xyz.reshape(B, S, H, W, 3)                                    # (B, S, H, W, 3)
    mask    = dedup_mask.to(device).to(dtype)                                    # (B, S, H, W)
    mask_b  = (mask > 0.5)                                                       # bool

    # Sanity: per-frame winner/loser ratios
    per_frame_winner = mask_b.float().mean(dim=(-1, -2))                         # (B, S)
    skip_frames = (per_frame_winner < min_winner_ratio_per_frame) | \
                  (per_frame_winner > 1 - min_loser_ratio_per_frame)             # (B, S)

    # Scene scale for normalization (detached)
    scene_scale = _scene_scale(pred_xyz.detach())

    losses = []
    n_pairs_total = 0

    # Pre-build winner xyz maps DETACHED (these are anchors)
    xyz_map_detached = xyz_map.detach()                                          # (B, S, H, W, 3)

    for b in range(B):
        for si in range(S):
            if skip_frames[b, si]:
                continue

            loser_mask_i = ~mask_b[b, si]                                        # (H, W) bool
            if loser_mask_i.sum() < 10:
                continue

            loser_idx = loser_mask_i.nonzero(as_tuple=False)                     # (Nl, 2): (y, x)
            Nl = loser_idx.shape[0]
            if Nl > max_loser_per_frame:
                # Random subsample to control compute
                perm = torch.randperm(Nl, device=device)[:max_loser_per_frame]
                loser_idx = loser_idx[perm]
                Nl = max_loser_per_frame

            # Loser xyz (with grad)
            loser_xyz = xyz_map[b, si, loser_idx[:, 0], loser_idx[:, 1], :]     # (Nl, 3)

            # Project loser_xyz into all OTHER frames
            other_indices = [sj for sj in range(S) if sj != si and not skip_frames[b, sj]]
            if not other_indices:
                continue

            # Build (1, S_other, 4, 4) and (1, S_other, 3, 3) for projection
            extr_other = extr_4x4[b:b+1, other_indices]                          # (1, S_o, 4, 4)
            intr_other = intr_3x3[b:b+1, other_indices]                          # (1, S_o, 3, 3)

            pixel_uv, depth_proj, valid_depth = _project_world_to_pixel(
                loser_xyz.unsqueeze(0),                                          # (1, Nl, 3)
                extr_other,
                intr_other,
            )
            # pixel_uv: (1, S_o, Nl, 2), valid_depth: (1, S_o, Nl)

            # Sample winner xyz at projected locations (DETACHED!)
            xyz_other_detached = xyz_map_detached[b:b+1, other_indices]          # (1, S_o, H, W, 3)
            sampled_xyz, valid_uv = _grid_sample_xyz(
                xyz_other_detached, pixel_uv, H, W,
            )
            # sampled_xyz: (1, S_o, Nl, 3), valid_uv: (1, S_o, Nl)

            # Determine if destination pixel is a winner (mask_j > 0.5)
            # Use nearest-int rounding on pixel_uv to query mask
            u_int = pixel_uv[..., 0].round().long().clamp(0, W - 1)              # (1, S_o, Nl)
            v_int = pixel_uv[..., 1].round().long().clamp(0, H - 1)              # (1, S_o, Nl)

            mask_other_b = mask_b[b:b+1, other_indices]                          # (1, S_o, H, W)
            S_o = len(other_indices)
            # Gather mask values at (v_int, u_int)
            # Flatten: mask_other_b → (1*S_o, H, W)
            mask_flat = mask_other_b.reshape(S_o, H, W)
            v_flat    = v_int.reshape(S_o, Nl)
            u_flat    = u_int.reshape(S_o, Nl)
            dest_is_winner = mask_flat[torch.arange(S_o, device=device).unsqueeze(1),
                                        v_flat, u_flat]                          # (S_o, Nl) bool
            dest_is_winner = dest_is_winner.unsqueeze(0)                         # (1, S_o, Nl)

            # Combine all validity flags
            valid_pair = valid_depth & valid_uv & dest_is_winner                 # (1, S_o, Nl)
            if valid_pair.sum() == 0:
                continue

            # Compute per-pair L1 distance, masked
            loser_xyz_exp = loser_xyz.unsqueeze(0).unsqueeze(1).expand_as(sampled_xyz)
            # (1, S_o, Nl, 3)
            diff = (loser_xyz_exp - sampled_xyz).abs().sum(dim=-1)               # (1, S_o, Nl)
            diff_n = diff / scene_scale                                          # normalize

            # Mask: only count valid pairs
            mask_f = valid_pair.float()
            n_valid = mask_f.sum().clamp_min(1.0)
            loss_i = (diff_n * mask_f).sum() / n_valid

            losses.append(loss_i)
            n_pairs_total += int(n_valid.item())

    if not losses:
        zero = pred_xyz.sum() * 0.0  # keeps the graph if needed
        return zero, {'n_pairs': 0, 'scene_scale': float(scene_scale.item())}

    total_loss = torch.stack(losses).mean()
    return total_loss, {
        'n_pairs': n_pairs_total,
        'scene_scale': float(scene_scale.item()),
        'n_frames_used': len(losses),
    }