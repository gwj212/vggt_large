# vggt/utils/frame_align.py
# ============================================================
# VGGT 预测坐标系  <->  CO3D GT 坐标系 的鲁棒相似变换 (Sim3)
#
# ★ 这个文件是修掉你日志里 "degenerate camera alignment (residual=...)" 的关键。
#
# 旧做法(病态): 对 3 个输入视角的相机中心做 Umeyama/lstsq 拟合 Sim3。
#   环拍取 3 帧时相机中心近似共线 -> 协方差矩阵秩亏 -> R 解不出来 ->
#   residual=nan 或巨大 -> 那一步被 skip; 更糟的是没被 skip 的那些步
#   相机同样是歪的, 补全头一直对着错位的目标训练。
#
# 新做法(良态): 分开解 R / s / c, 每一项都用最稳的量:
#   R : 由**相机朝向矩阵**求解, 不依赖相机中心分布。
#       同一台物理相机: R_pred_i = R_gt_i @ R  =>  R = R_gt_i^T @ R_pred_i
#       对所有 i 求和后 SVD 投影回 SO(3)。S=1 都能解, 共线也无所谓。
#   s : 相机中心两两距离之比的中位数 (需要 >=2 个不重合的中心)。
#   c : 质心对齐。
#
# 约定
# ----
#   extrinsic E = [R|t] (3x4), world->cam, OpenCV (x右 y下 z前):
#       X_cam = R @ X_world + t
#   相机中心 C = -R^T @ t
#
#   本文件求的相似变换定义为   X_gt = s * R @ X_pred + c
#
# 两种用法(推荐第一种)
# --------------------
#   ① 把 GT 相机搬进 VGGT 帧 (gt_extrinsics_to_pred_frame):
#      可见分支/深度/xyz_base 全部保持在 VGGT 原生帧不动 —— 冻结的重建权重
#      零风险, 补全头也在同一个帧里工作。留出视角用精确 GT 相机渲染。
#      对输入视角, 变换后的 GT 相机会**逐值退回** VGGT 预测的相机(对齐完美时),
#      这可以直接当自检 (见 check_roundtrip)。
#
#   ② 把高斯搬进 GT 帧 (transform_gaussians): 导出点云 / 与其他方法对比时用。
#
# 四元数约定: 本文件用 **wxyz (scalar-first)**, 与 gsplat / GaussianHead 的
#   rot_head(bias[0]=1.0) 一致。注意 vggt/utils/rotation.py 里的 mat_to_quat
#   用的是 xyzw (scalar-last), 两者不能混用 —— 这是很容易踩的坑。
# ============================================================

from typing import Dict, Optional, Tuple

import torch


# ------------------------------------------------------------
# 四元数 (wxyz) 工具
# ------------------------------------------------------------

def quat_wxyz_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(...,4) wxyz -> (...,3,3)。不要求输入已归一化。"""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) -> (...,4) wxyz。分支无关实现(取最大分量那一支, 数值最稳)。"""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    # 4 个候选的 |分量|*2
    q_abs = torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1).clamp_min(0.0).sqrt()

    cands = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1),
    ], dim=-2)                                            # (...,4,4)

    denom = (2.0 * q_abs).clamp_min(0.1)[..., None]        # 小分量那支不会被选中
    cands = cands / denom
    best = q_abs.argmax(dim=-1)                            # (...)
    q = torch.gather(cands, -2, best[..., None, None].expand(cands.shape[:-2] + (1, 4)))
    q = q.squeeze(-2)
    q = torch.where(q[..., 0:1] < 0, -q, q)                # 规范化到 w>=0
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """四元数乘 a*b, 都是 wxyz。对应旋转矩阵 R_a @ R_b。"""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


# ------------------------------------------------------------
# 基础几何
# ------------------------------------------------------------

def camera_centers(extr: torch.Tensor) -> torch.Tensor:
    """extr (...,3,4) -> 相机中心 (...,3)。C = -R^T t"""
    R = extr[..., :3, :3]
    t = extr[..., :3, 3]
    return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)


def project_to_so3(M: torch.Tensor) -> torch.Tensor:
    """把任意 3x3 投影到最近的旋转矩阵 (Frobenius 意义), 保证 det=+1。"""
    U, _, Vh = torch.linalg.svd(M.double())
    R = U @ Vh
    if torch.det(R) < 0:                                   # 反射 -> 翻最小奇异值方向
        U = U.clone()
        U[:, -1] = -U[:, -1]
        R = U @ Vh
    return R.to(M.dtype)


# ------------------------------------------------------------
# 核心: 拟合 Sim3
# ------------------------------------------------------------

def fit_pred_to_gt_similarity(
    pred_extr: torch.Tensor,          # (S,3,4)  VGGT 预测, world_pred -> cam
    gt_extr: torch.Tensor,            # (S,3,4)  CO3D GT,   world_gt   -> cam
    min_baseline_ratio: float = 1e-3,
) -> Tuple[float, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """返回 (s, R(3,3), c(3,), info)。满足 X_gt ≈ s * R @ X_pred + c。

    S 可以是 1(只能定 R, s 退化为 1)、2、或任意多。**不会因为相机共线而失败**。

    info 里的诊断量:
      rot_deg   : 各视角单独解出的 R 相对平均 R 的角度偏差(度)的中位数。
                  正常 < 3°; > 15° 说明 VGGT 位姿本身就崩了, 该步应当 skip。
      scale_cv  : 两两尺度比的离散度 (MAD/median)。正常 < 0.1。
      center_err: 对齐后相机中心的相对残差 (相对于场景半径)。正常 < 0.05。
    """
    assert pred_extr.shape == gt_extr.shape and pred_extr.shape[-2:] == (3, 4)
    dev = pred_extr.device
    S = pred_extr.shape[0]
    Rp = pred_extr[:, :3, :3].double()
    Rg = gt_extr[:, :3, :3].double()

    # ---- 1) 旋转: R = SO3proj( sum_i R_gt_i^T @ R_pred_i ) ----
    M = torch.einsum('sij,sik->jk', Rg, Rp)                # sum_i Rg_i^T Rp_i
    R = project_to_so3(M)

    # 诊断: 每个视角单独解出的 R_i 与平均 R 的夹角
    per_view = torch.einsum('sji,sjk->sik', Rg, Rp)        # R_i = Rg_i^T Rp_i
    rel = per_view @ R.transpose(-1, -2)
    cos = ((rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]) - 1.0) / 2.0
    rot_deg = torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))
    rot_deg_med = float(rot_deg.median().item()) if S > 0 else 0.0

    # ---- 2) 尺度: 相机中心两两距离比的中位数 ----
    Cp = camera_centers(pred_extr).double()
    Cg = camera_centers(gt_extr).double()
    scale_cv = 0.0
    if S >= 2:
        dp = torch.cdist(Cp, Cp)
        dg = torch.cdist(Cg, Cg)
        iu = torch.triu_indices(S, S, offset=1, device=dev)
        a, b = dp[iu[0], iu[1]], dg[iu[0], iu[1]]
        ref = a.max().clamp_min(1e-12)
        keep = a > (min_baseline_ratio * ref)              # 丢掉几乎重合的相机对
        if keep.any():
            ratios = (b[keep] / a[keep].clamp_min(1e-12))
            s = float(ratios.median().item())
            mad = float((ratios - ratios.median()).abs().median().item())
            scale_cv = mad / max(s, 1e-12)
        else:
            s = 1.0
    else:
        s = 1.0
    if not (s > 0) or s != s:                              # 兜底 nan/0
        s = 1.0

    # ---- 3) 平移: 质心对齐 ----
    c = Cg.mean(0) - s * (R @ Cp.mean(0))

    # 诊断: 对齐残差(相对场景半径)
    Cp_in_gt = s * (Cp @ R.transpose(-1, -2)) + c
    radius = (Cg - Cg.mean(0)).norm(dim=-1).max().clamp_min(1e-9)
    center_err = float(((Cp_in_gt - Cg).norm(dim=-1).mean() / radius).item())

    info = {'rot_deg': rot_deg_med, 'scale_cv': scale_cv,
            'center_err': center_err, 'scale': s, 'n_views': int(S)}
    return s, R.to(pred_extr.dtype), c.to(pred_extr.dtype), info


def gt_extrinsics_to_pred_frame(
    gt_extr: torch.Tensor,            # (N,3,4) 任意多视角的 GT 相机
    s: float,
    R: torch.Tensor,                  # (3,3)
    c: torch.Tensor,                  # (3,)
) -> torch.Tensor:
    """把 GT 相机搬进 VGGT 预测帧, 返回 (N,3,4)。

    推导: X_cam = R_g X_gt + t_g, 且 X_gt = s R X_pred + c
          => X_cam = (s R_g R) X_pred + (R_g c + t_g)
          整体缩放 s 不改变投影(齐次除法), 但高斯的 scale 是在 pred 帧的世界单位里,
          所以必须把外参也除以 s, 让 means 和 scales 处在同一套单位:
          E' = [ R_g R | (R_g c + t_g)/s ]
    对输入视角(对齐完美时) E' 会精确等于 VGGT 预测的外参 —— 见 check_roundtrip。
    """
    Rg = gt_extr[..., :3, :3]
    tg = gt_extr[..., :3, 3]
    R_new = Rg @ R
    t_new = (Rg @ c.unsqueeze(-1)).squeeze(-1) + tg
    t_new = t_new / max(s, 1e-12)
    return torch.cat([R_new, t_new.unsqueeze(-1)], dim=-1)


def gt_points_to_pred_frame(X_gt: torch.Tensor, s: float, R: torch.Tensor,
                            c: torch.Tensor) -> torch.Tensor:
    """X_pred = R^T (X_gt - c) / s。用于把 GT 帧的物体中心/半径搬进 pred 帧。"""
    return ((X_gt - c) @ R) / max(s, 1e-12)


def transform_gaussians(g: Dict[str, torch.Tensor], s: float, R: torch.Tensor,
                        c: torch.Tensor) -> Dict[str, torch.Tensor]:
    """把一组高斯从 pred 帧搬到 GT 帧 (导出点云 / 跨方法对比时用)。

    xyz   -> s R x + c
    scale -> s * scale        (各向同性缩放, 直接乘)
    rot   -> quat(R) * quat   (wxyz)
    其余 (opacity/color) 不变。
    """
    out = dict(g)
    out['xyz'] = s * (g['xyz'] @ R.transpose(-1, -2)) + c
    out['scale'] = g['scale'] * s
    qR = mat_to_quat_wxyz(R).to(g['rotation'].dtype).expand_as(g['rotation'])
    out['rotation'] = quat_mul_wxyz(qR, g['rotation'])
    return out


# ------------------------------------------------------------
# 自检
# ------------------------------------------------------------

def check_roundtrip(pred_extr: torch.Tensor, gt_extr: torch.Tensor
                    ) -> Dict[str, float]:
    """对输入视角做往返自检: 变换后的 GT 相机应当逼近 VGGT 预测的相机。

    返回 rot_deg / trans_rel。rot_deg 大 => VGGT 位姿本身崩了;
    trans_rel 大 => 尺度或平移解错了。训练脚本用它决定要不要 skip 这一步。
    """
    s, R, c, info = fit_pred_to_gt_similarity(pred_extr, gt_extr)
    back = gt_extrinsics_to_pred_frame(gt_extr, s, R, c)
    dR = back[:, :3, :3] @ pred_extr[:, :3, :3].transpose(-1, -2)
    cos = ((dR[:, 0, 0] + dR[:, 1, 1] + dR[:, 2, 2]) - 1.0) / 2.0
    rot_deg = float(torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).max().item())
    scene_scale = camera_centers(pred_extr).std(dim=0).norm().clamp_min(1e-9)
    trans_rel = float(((back[:, :3, 3] - pred_extr[:, :3, 3]).norm(dim=-1).max()
                       / scene_scale).item())
    return {'rot_deg': rot_deg, 'trans_rel': trans_rel, **info}