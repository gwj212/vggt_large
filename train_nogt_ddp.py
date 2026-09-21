#!/usr/bin/env python3
# train_nogt.py — 完全去 GT 的纯 geo_init 训练 (= v28 geo_init 路径剥掉所有 GT 机制)
#
# ============================================================
#  与 v28 (P1_MODE=geo_init) 的差异:
#
#  1. ★★★ 零 GT, 零数据预处理 ★★★
#     - 不跑 optimize_gaussians_multiview (老师), 不存 gt_gaussians / gt_opa_std。
#     - 训练图像每步直接 load_and_preprocess_images 按需加载, 没有 train cache。
#     - 没有 GT 质量过滤: find_scenes 出来的全部场景直接进训练 (坏场景靠 render
#       loss 鲁棒 + 失败 step 自动 skip 兜底)。
#
#  2. ★ 监督只有: geo_init 冷启动锚 (scale→点距 / opacity→常数) + render loss
#                + opacity 三项 (BCE / 熵 / 平衡)。无任何 param loss。
#
#  3. ★ 视角选择全程 random.sample (无 P1 锁 v0; 因为没有 GT 锚点要对齐)。
#     phase schedule 仅用于 geo-init 强度退火 (P1=1.0 / P2=0.3 / P3=0.05)。
#
#  4. ★ render loss 相机参数取自当次 forward 的 extr_pred/intr_pred (与高斯同系)。
#
#  保留 (与 v28 一致, 为公平对比):
#     - depth_unproject xyz_base, xyz_offset=0.3, warmup 500 / factor 0.1
#     - ColorHead 启用, dpt_feature_head 解冻并入 optimizer
#     - 所有 loss 权重 (W_RENDER=5.0, BCE=0.3, ENT=0.05, BAL=0.1)
#     - dedup_method='image', HARD_STOP_STEP=41600 (同 v28 算力预算)
#
#  DDP launch:
#     torchrun --standalone --nproc_per_node=4 /root/vggt/train_nogt.py
# ============================================================

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('VGGT_XYZ_BASE_SOURCE', 'depth_unproject')

import sys
import glob
import time
import math
import random
import datetime
import re
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from PIL import Image
from pathlib import Path

# ============================================================
# 配置
# ============================================================

TRAIN_DIR         = os.environ.get("TRAIN_DIR", "/data2/vggt/my_train_20v")
TEST_DIR          = os.environ.get("TEST_DIR", "/data2/vggt/my_test_multiview")
N_TEST_EVAL       = int(os.environ.get("N_TEST_EVAL", "20"))

EXP_TAG           = "nogt"
OUTPUT_DIR        = os.environ.get("OUTPUT_DIR", "/root/vggt/output/nogt_geoinit_pure_ddp")
RENDER_DIR        = os.path.join(OUTPUT_DIR, "renders")
# ★ 只有 test cache (轻量, 不含 GT); 训练数据无 cache。用独立目录避免与 scan 抢 cache_v1。
TEST_CACHE_DIR    = os.path.join(OUTPUT_DIR, "test_cache")

NCCL_TIMEOUT_MINUTES = 20

GH_FRAMES_CHUNK_SIZE = 1
GH_USE_CHECKPOINT    = True

SCENE_MIN_VIEWS   = 2
SCENE_MAX_VIEWS   = 20
SCENE_NUM_VIEWS   = 0
TRAIN_VIEWS_PER_STEP = int(os.environ.get("TRAIN_VIEWS_PER_STEP", "5"))

# ★ 训练图像长边上限 (0=不限, 与 v28 一致用到 518)。显存吃紧时设 448/392 降峰值,
#   下采样到 14 的倍数 (VGGT patch=14)。可用环境变量 MAX_IMG_SIDE 覆盖。
MAX_IMG_SIDE = int(os.environ.get('MAX_IMG_SIDE', '0'))

TEST_MIN_VIEWS    = 2
TEST_MAX_VIEWS    = int(os.environ.get("TEST_MAX_VIEWS", "8"))
TEST_NUM_VIEWS    = int(os.environ.get("TEST_NUM_VIEWS", "0"))

NUM_EPOCHS        = int(os.environ.get("NUM_EPOCHS", "10"))
LR                = 1e-4

# ★ 续训: ''/未设=从头训练; 'auto'=自动找 OUTPUT_DIR 里最新的 step ckpt; 或填具体 .pth 路径。
#   续训会恢复 模型 / optimizer / global_step / best 指标, 只有更高 PSNR / 更低 loss
#   才覆盖 best ckpt (重启不再一上来就把最佳 ckpt 冲掉)。用环境变量 RESUME 覆盖。
RESUME_CHECKPOINT = os.environ.get('RESUME', '').strip()

ENABLE_COLOR_HEAD     = True
COLOR_LR_MULTIPLIER   = 8.0

FREEZE_DPT_FEATURE_HEAD = os.environ.get("FREEZE_DPT_FEATURE_HEAD", "0") == "1"
FEATURE_HEAD_LR_MULT    = 0.5

XYZ_OFFSET_LOG_SCALE_OVERRIDE = math.log(0.3)
WARMUP_STEPS  = 500
WARMUP_FACTOR = 0.1
XYZ_LR_MULTIPLIER = 0.5

# render / opacity loss 权重 (同 v28)
W_RENDER          = 5.0
W_RENDER_L1       = 0.7
W_RENDER_SSIM     = 0.3
W_OPA_BCE         = 0.3
W_OPA_ENTROPY     = 0.05
W_OPA_BALANCE     = 0.1

# geo-init 冷启动锚 (同 v28 geo_init 默认)
GEO_SCALE_K   = float(os.environ.get('GEO_SCALE_K',   '1.0'))
GEO_OPA_CONST = float(os.environ.get('GEO_OPA_CONST', '0.5'))
W_GEO_SCALE   = float(os.environ.get('W_GEO_SCALE',   '1.0'))
W_GEO_OPA     = float(os.environ.get('W_GEO_OPA',     '0.5'))

DEDUP_METHOD              = 'image'
DEDUP_TIMEOUT_SEC         = 5.0
DEDUP_TIMING_LOG_EARLY    = 5
DEDUP_TIMING_LOG_INTERVAL = 200

PHASE2_START      = 5000
PHASE3_START      = 25000

LOG_INTERVAL      = int(os.environ.get("LOG_INTERVAL", "200"))
EVAL_INTERVAL     = int(os.environ.get("EVAL_INTERVAL", "2000"))
SAVE_INTERVAL     = int(os.environ.get("SAVE_INTERVAL", "10000"))

TEST_PATIENCE     = 20
MIN_TRAIN_STEPS   = int(os.environ.get("MIN_TRAIN_STEPS", "25000"))
HARD_STOP_STEP    = int(os.environ.get("HARD_STOP_STEP", "41600"))

VGGT_MODEL        = os.environ.get("VGGT_MODEL", "/root/models/VGGT-1B")
LOW_VRAM_MODE     = os.environ.get("LOW_VRAM_MODE", "0") == "1"
SKIP_TEST_PRECOMPUTE = os.environ.get("SKIP_TEST_PRECOMPUTE", "0") == "1"
MAX_TRAIN_SCENES  = int(os.environ.get("MAX_TRAIN_SCENES", "0"))
SANITY_VIEWS      = int(os.environ.get("SANITY_VIEWS", "5"))
LOG_ALL_RANK_SHAPES = os.environ.get("LOG_ALL_RANK_SHAPES", "0") == "1"

PSNR_SANE_MAX     = 55.0

sys.path.insert(0, "/root/vggt")

from vggt.models.vggt import VGGT, unproject_depth_to_world_torch
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.gs_optimizer import render_gaussians, ssim
from vggt.losses.dedup_mask import compute_dedup_mask


# ============================================================
# 日志: stdout/stderr 同时写入 .log 文件 (所有 print/log_* 自动落盘)
# ============================================================
class _Tee:
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data); s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def setup_file_logging(log_dir, tag):
    """把 sys.stdout/stderr tee 到 <log_dir>/<tag>[_rankN].log (固定名, 跨次追加不覆盖)。
       每次启动打一条 RUN START 横幅分隔; 旧的带时间戳日志保留作历史。"""
    # RANK is global across nodes; LOCAL_RANK repeats on every node.
    rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))
    os.makedirs(log_dir, exist_ok=True)
    suffix = '' if str(rank) == '0' else f'_rank{rank}'
    path = os.path.join(log_dir, f'{tag}{suffix}.log')      # 固定名 → 续训接着写
    fh = open(path, 'a', buffering=1)                       # append + 行缓冲
    sys.stdout = _Tee(sys.__stdout__ or sys.stdout, fh)
    sys.stderr = _Tee(sys.__stderr__ or sys.stderr, fh)
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "#" * 70, flush=True)
    print(f"#  RUN START {ts}   (日志追加写入, 不覆盖历史)", flush=True)
    print("#" * 70, flush=True)
    print(f"[log] 控制台输出同时写入(追加): {path}", flush=True)
    return path


# ============================================================
# 分布式 (与 v28 相同)
# ============================================================

def setup_ddp():
    if 'LOCAL_RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        raise RuntimeError(
            "train_nogt.py is DDP-only; launch it with torchrun --nproc_per_node=<N>"
        )
    dist.init_process_group(backend='nccl',
                            timeout=datetime.timedelta(minutes=NCCL_TIMEOUT_MINUTES))
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank(); world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def log_main(rank, *a, **k):
    if rank == 0:
        print(*a, **k, flush=True)


def log_all(rank, *a, **k):
    print(f"[rank {rank}]", *a, **k, flush=True)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def unwrap_ddp(module):
    return module.module if isinstance(module, DDP) else module


# ============================================================
# LPIPS / 指标 / 工具 (与 v28 相同)
# ============================================================

_lpips_fn = None

def get_lpips_fn(device):
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips
            _lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
            for p in _lpips_fn.parameters():
                p.requires_grad_(False)
        except ImportError:
            _lpips_fn = "unavailable"
    return _lpips_fn


def compute_lpips(pred, gt, device):
    fn = get_lpips_fn(device)
    if fn == "unavailable":
        return -1.0
    with torch.no_grad():
        p = pred.unsqueeze(0).float().to(device) * 2 - 1
        g = gt.unsqueeze(0).float().to(device) * 2 - 1
        return fn(p, g).item()


def copy_point_head_to_feature_head(model):
    assert model.point_head is not None and model.dpt_feature_head is not None
    ph, fh = model.point_head.state_dict(), model.dpt_feature_head.state_dict()
    copied = 0
    for name, param in ph.items():
        if name in fh and param.shape == fh[name].shape:
            fh[name].copy_(param); copied += 1
    model.dpt_feature_head.load_state_dict(fh)
    return copied


def find_images(directory):
    exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    imgs = []
    for e in exts:
        imgs.extend(glob.glob(os.path.join(directory, '**', e), recursive=True))
    imgs.sort()
    return imgs


def find_scenes(directory, min_views=2, max_views=20, num_views=0):
    scene_dirs = sorted(d for d in glob.glob(os.path.join(directory, "*")) if os.path.isdir(d))
    scenes = []
    for sd in scene_dirs:
        imgs = find_images(sd)
        if len(imgs) < min_views:
            continue
        n = min(num_views, len(imgs)) if num_views > 0 else min(max_views, len(imgs))
        if n < len(imgs):
            idx = np.linspace(0, len(imgs) - 1, n, dtype=int)
            imgs = [imgs[i] for i in idx]
        scenes.append(imgs)
    return scenes


def format_duration(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    if h > 0: return f"{h}h {m}m {sec}s"
    if m > 0: return f"{m}m {sec}s"
    return f"{sec}s"


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred.float(), gt.float()).item()
    if mse < 1e-10 or math.isnan(mse):
        return 60.0
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim_val(pred, gt, window_size=11):
    p = pred.float().unsqueeze(0); g = gt.float().unsqueeze(0)
    C1, C2 = 0.01**2, 0.03**2
    ch = p.shape[1]; sigma = 1.5
    coords = torch.arange(window_size, dtype=p.dtype, device=p.device) - window_size // 2
    k1d = torch.exp(-(coords**2) / (2 * sigma**2)); k1d /= k1d.sum()
    kernel = k1d.outer(k1d).unsqueeze(0).unsqueeze(0).repeat(ch, 1, 1, 1)
    pad = window_size // 2
    mu1 = F.conv2d(p, kernel, padding=pad, groups=ch)
    mu2 = F.conv2d(g, kernel, padding=pad, groups=ch)
    m1sq, m2sq, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(p * p, kernel, padding=pad, groups=ch) - m1sq
    s2 = F.conv2d(g * g, kernel, padding=pad, groups=ch) - m2sq
    s12 = F.conv2d(p * g, kernel, padding=pad, groups=ch) - m12
    ssim_map = ((2 * m12 + C1) * (2 * s12 + C2)) / ((m1sq + m2sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean().item()


def save_render_comparison(rendered, gt, step, psnr, ssim_val, save_dir, prefix="", lpips_val=None):
    os.makedirs(save_dir, exist_ok=True)
    def to_pil(t):
        arr = (t.detach().cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr.transpose(1, 2, 0))
    r, g_img = to_pil(rendered), to_pil(gt)
    W_img, H_img = r.width, r.height
    canvas = Image.new("RGB", (W_img * 2 + 4, H_img), (128, 128, 128))
    canvas.paste(r, (0, 0)); canvas.paste(g_img, (W_img + 4, 0))
    ls = f"_lpips{lpips_val:.4f}" if lpips_val is not None and lpips_val >= 0 else ""
    fname = f"{prefix}step{step:06d}_psnr{psnr:.2f}_ssim{ssim_val:.4f}{ls}.png"
    canvas.save(os.path.join(save_dir, fname))
    return fname


# ============================================================
# Loss (无任何 GT param loss; 只有 geo-init / render / opacity)
# ============================================================
_EPS = 1e-6


def geo_init_param_loss(pred_gaussians, xyz_base, S_sel, H, W,
                        scale_k, opa_const, w_scale, w_opa, strength):
    """去GT 冷启动锚 (纯几何): scale→scale_k×局部点距(log空间); opacity→常数。
       对所有选中帧的 per-pixel 高斯施加; strength = phase 退火。目标全程 detach。"""
    scale = pred_gaussians['scale'][0, :S_sel]
    opa = pred_gaussians['opacity'][0, :S_sel]
    with torch.no_grad():
        xb = xyz_base[0, :S_sel].reshape(S_sel, H, W, 3).float()
        dr = (xb[:, :, 1:, :] - xb[:, :, :-1, :]).norm(dim=-1)
        dd = (xb[:, 1:, :, :] - xb[:, :-1, :, :]).norm(dim=-1)
        spacing = torch.zeros(S_sel, H, W, device=xb.device)
        cnt = torch.zeros(S_sel, H, W, device=xb.device)
        spacing[:, :, 1:] += dr; cnt[:, :, 1:] += 1
        spacing[:, :, :-1] += dr; cnt[:, :, :-1] += 1
        spacing[:, 1:, :] += dd; cnt[:, 1:, :] += 1
        spacing[:, :-1, :] += dd; cnt[:, :-1, :] += 1
        spacing = (spacing / cnt.clamp(min=1)).reshape(S_sel, H * W, 1).clamp(1e-6)
        target_log_scale = torch.log((scale_k * spacing).clamp(1e-8))
    l_scale = (torch.log(scale.clamp(1e-8)) - target_log_scale).abs().mean()
    l_opa = ((opa - opa_const) ** 2).mean()
    total = strength * (w_scale * l_scale + w_opa * l_opa)
    return total, float(l_scale.item()), float(l_opa.item())


def opacity_bce_loss(pred_opacity, dedup_mask):
    B, S, HW, _ = pred_opacity.shape
    target = dedup_mask.reshape(B, S, HW, 1)
    # ★ 消毒: dedup_mask 在退化场景可能含 NaN/Inf/越界值, 直接喂 BCE 会触发
    #   device-side assert (target 必须在 [0,1])。先 nan_to_num + clamp。
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return F.binary_cross_entropy(pred_opacity.clamp(_EPS, 1 - _EPS), target)


def opacity_entropy_loss(pred_opacity):
    a = pred_opacity.clamp(_EPS, 1 - _EPS)
    return (-(a * a.log() + (1 - a) * (1 - a).log())).mean()


def opacity_balance_loss(pred_opacity, k=20.0):
    soft_surf = torch.sigmoid((pred_opacity - 0.5) * k).sum(dim=2).squeeze(-1)
    target = soft_surf.mean(dim=1, keepdim=True)
    return ((soft_surf - target) ** 2 / (target.pow(2) + _EPS)).mean()


def compute_render_loss(pred_frame, gt_image, ext, itr, H, W):
    rendered = render_gaussians(
        xyz=pred_frame['xyz'].float(), rotation=pred_frame['rotation'].float(),
        scale=pred_frame['scale'].float(), opacity=pred_frame['opacity'].float(),
        color=pred_frame['color'].float(), extrinsics=ext, intrinsics=itr, H=H, W=W)
    l1 = F.l1_loss(rendered, gt_image)
    sloss = 1.0 - ssim(rendered.unsqueeze(0), gt_image.unsqueeze(0))
    return W_RENDER_L1 * l1 + W_RENDER_SSIM * sloss, rendered


def build_param_groups(gh, base_lr, xyz_lr_mult=1.0, color_lr_mult=1.0):
    geo, xyz, color = [], [], []
    for name, p in gh.named_parameters():
        if not p.requires_grad:
            continue
        if 'color_head' in name:
            color.append(p)
        elif 'xyz_head' in name or 'xyz_offset' in name:
            xyz.append(p)
        else:
            geo.append(p)
    groups = [{'params': geo, 'lr': base_lr, 'name': 'geo'},
              {'params': xyz, 'lr': base_lr * xyz_lr_mult, 'name': 'xyz'}]
    if color:
        groups.append({'params': color, 'lr': base_lr * color_lr_mult, 'name': 'color'})
    return groups


def safe_compute_dedup_mask(method, xyz_base, depth_conf, extr, intr, H, W,
                            global_step, timeout_sec=DEDUP_TIMEOUT_SEC, log_timing=False, rank=0):
    t = time.time()
    try:
        mask = compute_dedup_mask(method=method, xyz_base=xyz_base, depth_conf=depth_conf,
                                  extrinsics=extr, intrinsics=intr, H=H, W=W)
        el = time.time() - t
        if log_timing:
            log_all(rank, f"  [mask] step{global_step} method={method} 耗时={el:.2f}s ratio={mask.mean().item():.2f}")
        return mask, el, None
    except Exception as e:
        el = time.time() - t
        log_all(rank, f"  ⚠️ [mask] step{global_step} {method} 失败: {e}, 退化为 'none'")
        B, S, _, _, _ = xyz_base.shape
        return torch.ones(B, S, H, W, device=xyz_base.device), el, str(e)


# ============================================================
# 测试集预计算 (轻量, 无 GT; 与 v28 相同)
# ============================================================

def precompute_test_scene_light(model, image_paths, device, dtype, cache_path):
    if os.path.exists(cache_path):
        try:
            d = torch.load(cache_path, map_location='cpu')
            assert all(k in d for k in ('gt_images', 'images_input', 'exts', 'itrs', 'num_views'))
            return d
        except Exception:
            os.remove(cache_path)
    images = load_and_preprocess_images(image_paths).to(device)
    images_input = images.unsqueeze(0)
    S, _, H, W = images.shape
    model.eval()
    _gh, _dfh = model.gaussian_head, model.dpt_feature_head
    model.gaussian_head, model.dpt_feature_head = None, None
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                preds = model(images_input)
    finally:
        model.gaussian_head, model.dpt_feature_head = _gh, _dfh
    extr, intr = pose_encoding_to_extri_intri(preds['pose_enc'], image_size_hw=(H, W))
    d = {'scene_name': Path(image_paths[0]).parent.name, 'image_paths': image_paths,
         'num_views': S, 'gt_images': [images[s].float().cpu() for s in range(S)],
         'exts': [extr[0, s].float().cpu() for s in range(S)],
         'itrs': [intr[0, s].float().cpu() for s in range(S)],
         'H': H, 'W': W, 'images_input': images_input.cpu()}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(d, cache_path)
    del preds, images_input, images
    torch.cuda.empty_cache()
    return d


def precompute_test_split(model, my_scenes, device, dtype, cache_dir, rank, idx_map):
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    for li, scene_imgs in enumerate(my_scenes):
        gi = idx_map[li]
        name = Path(scene_imgs[0]).parent.name
        cp = os.path.join(cache_dir, f"test_{gi:05d}_{name}.pt")
        try:
            precompute_test_scene_light(model, scene_imgs, device, dtype, cp)
        except Exception as e:
            log_all(rank, f"  [precomp test] {name} 失败: {e}")
            torch.cuda.empty_cache()
    log_all(rank, f"  [precomp test] DONE: {len(my_scenes)} 场景 | {format_duration(time.time()-t0)}")


# ============================================================
# 评估 (与 v28 相同)
# ============================================================

def evaluate_single_test_scene(model, td, device, dtype, viz_idx=None):
    images_input = td['images_input'].to(device)
    S, H, W = td['num_views'], td['H'], td['W']
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            pe = model(images_input)
        ps = {k: v[0].reshape(-1, v.shape[-1]).float() for k, v in pe['gaussians'].items()
              if isinstance(v, torch.Tensor) and v.dim() >= 4}
        psnrs, ssims, lpipss, viz = [], [], [], {}
        viz_set = set(viz_idx) if viz_idx else set()
        use_merged = True
        for s in range(S):
            ext = td['exts'][s].to(device); itr = td['itrs'][s].to(device); gt = td['gt_images'][s].to(device)
            try:
                if use_merged:
                    rd = render_gaussians(ps['xyz'], ps['rotation'], ps['scale'], ps['opacity'], ps['color'], ext, itr, H, W)
                else:
                    ef = {k: v[0, s].float() for k, v in pe['gaussians'].items() if isinstance(v, torch.Tensor) and v.dim() >= 4}
                    rd = render_gaussians(ef['xyz'], ef['rotation'], ef['scale'], ef['opacity'], ef['color'], ext, itr, H, W)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and use_merged:
                    torch.cuda.empty_cache(); use_merged = False
                    try:
                        ef = {k: v[0, s].float() for k, v in pe['gaussians'].items() if isinstance(v, torch.Tensor) and v.dim() >= 4}
                        rd = render_gaussians(ef['xyz'], ef['rotation'], ef['scale'], ef['opacity'], ef['color'], ext, itr, H, W)
                    except Exception:
                        torch.cuda.empty_cache(); continue
                else:
                    torch.cuda.empty_cache(); continue
            p_s = compute_psnr(rd, gt); s_s = compute_ssim_val(rd, gt); lp = compute_lpips(rd, gt, device)
            psnrs.append(p_s); ssims.append(s_s)
            if lp >= 0:
                lpipss.append(lp)
            if s in viz_set:
                viz[s] = (rd, gt, p_s, s_s, lp)
            torch.cuda.empty_cache()
    return (sum(psnrs) / max(len(psnrs), 1), sum(ssims) / max(len(ssims), 1),
            sum(lpipss) / len(lpipss) if lpipss else -1.0, viz)


def evaluate_on_test_set(model, tds, device, dtype, step, render_dir, save_renders=3):
    model.eval()
    psnrs, ssims, lpipss = [], [], []
    n_bad = 0
    for i, td in enumerate(tds):
        viz_views = sorted(set([0, td['num_views'] // 2, td['num_views'] - 1])) if i < save_renders else None
        try:
            p, s, lp, viz = evaluate_single_test_scene(model, td, device, dtype, viz_views)
            if not math.isfinite(p) or p > PSNR_SANE_MAX:
                n_bad += 1; continue
            psnrs.append(p); ssims.append(s)
            if lp >= 0:
                lpipss.append(lp)
            if viz_views and viz:
                for v in viz_views:
                    if v not in viz:
                        continue
                    rd, gt, pp, ss, ll = viz[v]
                    save_render_comparison(rd, gt, step, pp, ss, render_dir, prefix=f"scene{i:02d}_v{v}_", lpips_val=ll)
        except Exception:
            pass
        torch.cuda.empty_cache()
    if not psnrs:
        return 0.0, 0.0, -1.0, "评估失败"
    extra = f" (跳过异常 {n_bad})" if n_bad else ""
    return (sum(psnrs) / len(psnrs), sum(ssims) / len(ssims),
            sum(lpipss) / len(lpipss) if lpipss else -1.0,
            f"avg over {len(psnrs)} scenes{extra} | PSNR [{min(psnrs):.1f}, {max(psnrs):.1f}]")


# ============================================================
# Sanity check (从原图加载, 无 cache)
# ============================================================

def sanity_check_xyz_base(model, scene_imgs, device, dtype, rank):
    if rank != 0:
        return
    log_main(rank, "\n" + "=" * 70)
    log_main(rank, "  [Sanity Check] xyz_base + dedup_mask 健康检查 (从原图)")
    log_main(rank, "=" * 70)
    try:
        n = min(SANITY_VIEWS, len(scene_imgs))
        images_input, _, _, H, W = load_selected_views(scene_imgs, n, device)
        images = images_input[0]
    except Exception as e:
        log_main(rank, f"  ⚠️ 加载失败: {e}"); return
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = model(images_input)
    if not all(k in preds for k in ('world_points', 'depth', 'pose_enc')):
        log_main(rank, "  ⚠️ predictions 字段不全, 跳过"); return
    old_xyz = preds['world_points'].float()
    extr, intr = pose_encoding_to_extri_intri(preds['pose_enc'], image_size_hw=(H, W))
    new_xyz = unproject_depth_to_world_torch(preds['depth'].float(), extr.float(), intr.float())
    rel = (old_xyz - new_xyz).norm(dim=-1).mean() / (old_xyz.norm(dim=-1).mean() + 1e-8)
    log_main(rank, f"  scene name        : {Path(scene_imgs[0]).parent.name}")
    log_main(rank, f"  ★ relative diff   : {rel.item()*100:.1f}% of scene scale")
    try:
        xyz_base = unproject_depth_to_world_torch(preds['depth'].float(), extr.float(), intr.float())
        mask = compute_dedup_mask(method=DEDUP_METHOD, xyz_base=xyz_base,
                                  depth_conf=preds['depth_conf'].float(),
                                  extrinsics=extr.float(), intrinsics=intr.float(), H=H, W=W)
        pf = mask[0].mean(dim=(1, 2)).cpu().numpy()
        log_main(rank, f"  dedup mask ratio  : {mask.mean().item()*100:.1f}% | "
                       f"per-frame [{','.join(f'{x*100:.0f}%' for x in pf)}] fstd {float(pf.std())*100:.2f}%")
    except Exception as e:
        log_main(rank, f"  ⚠️ dedup_mask 检查失败: {e}")
    log_main(rank, "=" * 70 + "\n")
    del preds, old_xyz, new_xyz, images_input, images
    torch.cuda.empty_cache()


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(model, optimizer, step, epoch, loss_val, path, metrics=None, best_state=None):
    ckpt = {'global_step': step, 'epoch': epoch, 'loss': loss_val,
            'gaussian_head_state_dict': unwrap_ddp(model.gaussian_head).state_dict(),
            'dpt_feature_head_state_dict': unwrap_ddp(model.dpt_feature_head).state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
    if metrics is not None:
        ckpt['metrics'] = metrics
    if best_state is not None:                  # ★ 续训追踪: 把当前 best 指标一起存进去
        ckpt['best_state'] = best_state
    torch.save(ckpt, path)


def _best_state(best_psnr, best_ssim, best_lpips, best_loss):
    return {'best_psnr': best_psnr, 'best_ssim': best_ssim,
            'best_lpips': best_lpips, 'best_loss': best_loss}


def find_resume_checkpoint(output_dir, tag):
    """RESUME='auto' 时挑一个 ckpt 续训: 优先 step 最大的; 否则 best_loss; 再 best_test。"""
    step_ckpts = glob.glob(os.path.join(output_dir, f"gaussian_head_step*_{tag}.pth"))
    if step_ckpts:
        def _stepnum(p):
            m = re.search(r'step(\d+)_', os.path.basename(p))
            return int(m.group(1)) if m else -1
        return max(step_ckpts, key=_stepnum)
    for name in (f"gaussian_head_best_loss_{tag}.pth", f"gaussian_head_best_test_{tag}.pth"):
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_checkpoint_for_resume(model, optimizer, ckpt_path, output_dir, tag, device, rank):
    """加载 模型 + optimizer 状态, 返回 (next_step, best_psnr, best_ssim, best_lpips, best_loss)。
       - next_step = ckpt 里的 global_step + 1 (该 step 已完成, 从下一步接着跑)。
       - best 指标优先取 ckpt['best_state'] (新格式); 旧 ckpt 无此字段时, 从同目录的
         best_test ckpt['metrics'] 和 best_loss ckpt['loss'] 兜底, 以保护既有最佳 ckpt
         不被续训第一次 eval 覆盖。
       - optimizer 状态从 CPU 读入后, 张量搬到 device (否则 Adam 会 device 不一致报错)。"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.gaussian_head.load_state_dict(ckpt['gaussian_head_state_dict'])
    if 'dpt_feature_head_state_dict' in ckpt:
        model.dpt_feature_head.load_state_dict(ckpt['dpt_feature_head_state_dict'])
    if 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception as e:
            log_main(rank, f"  ⚠️ optimizer 状态加载失败 ({e}), 用全新 optimizer 继续")

    next_step = int(ckpt.get('global_step', 0)) + 1

    bs = ckpt.get('best_state', {})
    best_psnr  = float(bs.get('best_psnr', 0.0))
    best_ssim  = float(bs.get('best_ssim', 0.0))
    best_lpips = float(bs.get('best_lpips', float('inf')))
    best_loss  = float(bs.get('best_loss', float('inf')))

    if not bs:    # 旧 ckpt: 从既有 best_test / best_loss ckpt 兜底, 保护它们
        bt = os.path.join(output_dir, f"gaussian_head_best_test_{tag}.pth")
        if os.path.exists(bt):
            try:
                m = torch.load(bt, map_location='cpu').get('metrics', {})
                if 'psnr' in m:
                    best_psnr = max(best_psnr, float(m['psnr']))
                if 'ssim' in m:
                    best_ssim = float(m['ssim'])
                if m.get('lpips', -1) is not None and float(m.get('lpips', -1)) >= 0:
                    best_lpips = float(m['lpips'])
            except Exception:
                pass
        bl = os.path.join(output_dir, f"gaussian_head_best_loss_{tag}.pth")
        if os.path.exists(bl):
            try:
                lv = torch.load(bl, map_location='cpu').get('loss', None)
                if lv is not None:
                    best_loss = min(best_loss, float(lv))
            except Exception:
                pass

    for st in optimizer.state.values():          # 把 optimizer state 张量搬到 GPU
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device)

    return next_step, best_psnr, best_ssim, best_lpips, best_loss


# ============================================================
# 训练数据: 每步从原图按需加载 (无 cache, 无 GT)
# ============================================================

def _round14(x):
    return max(14, int(round(x / 14)) * 14)


def load_selected_views(scene_imgs, n_pick, device):
    """随机选 n_pick 个视角, 只加载这几张图 (避免 H/W 跨步不一致的全集加载在长跑里的开销)。
       返回 (images_input[1,n,3,H,W] on device, gt_images_sel list, selected, H, W)。"""
    S_total = len(scene_imgs)
    n = min(n_pick, S_total)
    selected = random.sample(range(S_total), n)
    sel_paths = [scene_imgs[s] for s in selected]
    imgs = load_and_preprocess_images(sel_paths)            # CPU [n,3,H,W]
    n, _, H, W = imgs.shape
    # ★ 可选下采样降显存峰值 (保持 14 的倍数; 在 CPU 上做, 不占 GPU)
    if MAX_IMG_SIDE > 0 and max(H, W) > MAX_IMG_SIDE:
        scale = MAX_IMG_SIDE / max(H, W)
        nH, nW = _round14(H * scale), _round14(W * scale)
        try:
            imgs = F.interpolate(imgs, size=(nH, nW), mode='bilinear',
                                 align_corners=False, antialias=True)
        except TypeError:                                   # 老版本 torch 无 antialias
            imgs = F.interpolate(imgs, size=(nH, nW), mode='bilinear', align_corners=False)
        H, W = nH, nW
    imgs = imgs.to(device)
    images_input = imgs.unsqueeze(0)
    gt_images_sel = [imgs[i].float() for i in range(n)]
    return images_input, gt_images_sel, selected, H, W


# ============================================================
# 主流程
# ============================================================

def main():
    log_path = setup_file_logging(OUTPUT_DIR, EXP_TAG)
    total_start = time.time()
    local_rank, rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}"
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability(local_rank)[0] >= 8
             else torch.float16)

    PHASE2_LOCAL = max(1, PHASE2_START // world_size)
    PHASE3_LOCAL = max(1, PHASE3_START // world_size)
    MIN_TRAIN_LOCAL = max(1, MIN_TRAIN_STEPS // world_size)
    HARD_STOP_LOCAL = max(1, HARD_STOP_STEP // world_size)
    EVAL_LOCAL = max(1, EVAL_INTERVAL // world_size)
    SAVE_LOCAL = max(1, SAVE_INTERVAL // world_size)
    LOG_LOCAL = max(1, LOG_INTERVAL // world_size)

    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(RENDER_DIR, exist_ok=True)
    barrier()

    log_main(rank, "=" * 70)
    log_main(rank, "  VGGT GaussianHead 训练  [nogt] 完全去 GT + 无数据预处理")
    log_main(rank, "  ★ 监督 = geo-init 锚 (scale→点距 / opa→常数) + render loss + opacity三项")
    log_main(rank, "  ★ 无 GT param loss / 无 gt_gaussians / 无 gt_opa_std 筛选")
    log_main(rank, "  ★ 训练图像每步从原图按需加载 (无 train cache)")
    log_main(rank, f"  ★ 视角全程 random.sample({TRAIN_VIEWS_PER_STEP}); phase 仅退火 geo 强度 P1=1.0/P2=0.3/P3=0.05")
    log_main(rank, f"  ★ geo: scale_k={GEO_SCALE_K} opa={GEO_OPA_CONST} w_scale={W_GEO_SCALE} w_opa={W_GEO_OPA}")
    log_main(rank, f"  ★ Loss: RENDER={W_RENDER} BCE={W_OPA_BCE} ENT={W_OPA_ENTROPY} BAL={W_OPA_BALANCE}")
    log_main(rank, f"  ★ HARD_STOP_STEP = {HARD_STOP_STEP} (与 v28 同算力预算)")
    log_main(rank, f"  Output: {OUTPUT_DIR} | world_size={world_size} | dtype={dtype}")
    log_main(rank, "=" * 70)

    # ---- 1. 模型 ----
    log_main(rank, "\n[1/4] 加载 VGGT-1B...")
    t0 = time.time()
    model = VGGT.from_pretrained(VGGT_MODEL, enable_gaussian=True)
    n_copied = None
    if LOW_VRAM_MODE:
        # Keep trainable reconstruction parameters FP32. The frozen backbone
        # and feature extractor can safely use FP16; the camera head runs on CPU.
        n_copied = copy_point_head_to_feature_head(model)
        model.aggregator.half()
        model.dpt_feature_head.half()
        model.point_head = None
        model.track_head = None
        model.aggregator.to(device)
        model.depth_head.to(device)
        model.dpt_feature_head.to(device)
        model.gaussian_head.to(device)
        model.camera_head.cpu()
    else:
        model.to(device)
    if hasattr(model.gaussian_head, 'frames_chunk_size'):
        model.gaussian_head.frames_chunk_size = GH_FRAMES_CHUNK_SIZE
        model.gaussian_head.use_checkpoint = GH_USE_CHECKPOINT
    if ENABLE_COLOR_HEAD:
        if hasattr(model.gaussian_head, 'enable_color_head_after_init'):
            model.gaussian_head.enable_color_head_after_init(device=device)
            log_main(rank, "  ✓ ColorHead 已启用")
        elif getattr(model.gaussian_head, 'color_head', None) is not None:
            log_main(rank, "  ✓ ColorHead 已默认启用")
    if n_copied is None:
        n_copied = copy_point_head_to_feature_head(model)
    log_main(rank, f"  ✓ 复制 {n_copied} 个张量 point_head→dpt_feature_head [解冻]")
    if LOW_VRAM_MODE:
        torch.cuda.empty_cache()
        log_main(rank, "  ✓ LOW_VRAM_MODE: FP16 backbone/frozen DPT, FP32 GaussianHead, CPU camera, disabled point/track heads")
    for n, p in model.named_parameters():
        if 'gaussian_head' in n:
            p.requires_grad_(True)
        elif 'dpt_feature_head' in n:
            p.requires_grad_(not FREEZE_DPT_FEATURE_HEAD)
        else:
            p.requires_grad_(False)
    log_main(rank, f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if rank == 0 and not SKIP_TEST_PRECOMPUTE:
        get_lpips_fn(device)
    log_main(rank, f"  耗时: {format_duration(time.time()-t0)}")

    # ---- 2. 场景列表 (无 GT 筛选, 全部进训练) ----
    log_main(rank, "\n[2/4] 扫描训练场景 (无筛选, 无 cache)...")
    scenes = find_scenes(TRAIN_DIR, min_views=SCENE_MIN_VIEWS, max_views=SCENE_MAX_VIEWS, num_views=SCENE_NUM_VIEWS)
    if MAX_TRAIN_SCENES > 0:
        scenes = scenes[:MAX_TRAIN_SCENES]
    assert len(scenes) > 0
    log_main(rank, f"  训练场景: {len(scenes)} | 平均 {sum(len(s) for s in scenes)/len(scenes):.1f} views/scene "
                   f"(全部直接进训练, 坏场景靠 render 鲁棒 + 失败 skip)")

    # ---- 3. 测试集预计算 (轻量) ----
    log_main(rank, "\n[3/4] 测试集预计算 (轻量, 无 GT)...")
    if SKIP_TEST_PRECOMPUTE:
        test_scenes = []
        log_main(rank, "  SKIP_TEST_PRECOMPUTE=1: smoke run skips test cache/evaluation")
    else:
        test_scenes = find_scenes(TEST_DIR, min_views=TEST_MIN_VIEWS, max_views=TEST_MAX_VIEWS, num_views=TEST_NUM_VIEWS)
        assert len(test_scenes) > 0
        my_test = test_scenes[rank::world_size]
        my_test_idx = list(range(rank, len(test_scenes), world_size))
        barrier()
        precompute_test_split(model, my_test, device, dtype, TEST_CACHE_DIR, rank, my_test_idx)
        barrier()

    eval_test_data, all_test_data = None, None
    if rank == 0:
        all_test_data = []
        for idx in range(len(test_scenes)):
            name = Path(test_scenes[idx][0]).parent.name
            cp = os.path.join(TEST_CACHE_DIR, f"test_{idx:05d}_{name}.pt")
            if os.path.exists(cp):
                try:
                    all_test_data.append(torch.load(cp, map_location='cpu'))
                except Exception:
                    pass
        if len(all_test_data) > N_TEST_EVAL:
            random.Random(42).shuffle(all_test_data)
            eval_test_data = sorted(all_test_data[:N_TEST_EVAL], key=lambda d: d.get('scene_name', ''))
        else:
            eval_test_data = all_test_data
        log_main(rank, f"  test cache: {len(all_test_data)} | 评估子集 {len(eval_test_data)}")
    barrier()

    # ---- 4. 初始化 optimizer + xyz_offset + sanity ----
    log_main(rank, "\n[4/4] 初始化 + xyz_offset / Sanity (from scratch)...")
    gh = model.gaussian_head
    param_groups = build_param_groups(gh, LR, xyz_lr_mult=XYZ_LR_MULTIPLIER, color_lr_mult=COLOR_LR_MULTIPLIER)
    if not FREEZE_DPT_FEATURE_HEAD:
        fh_params = [p for p in model.dpt_feature_head.parameters() if p.requires_grad]
        if fh_params:
            param_groups.append({'params': fh_params, 'lr': LR * FEATURE_HEAD_LR_MULT, 'name': 'feat'})
            log_main(rank, f"  ★ dpt_feature_head 入 optimizer: {sum(p.numel() for p in fh_params):,} 参数, lr={LR*FEATURE_HEAD_LR_MULT:.1e}")
    optimizer = optim.Adam(param_groups, weight_decay=1e-5)

    # ---- 续训: 解析要恢复的 ckpt ----
    resume_path = None
    if RESUME_CHECKPOINT:
        if RESUME_CHECKPOINT.lower() == 'auto':
            resume_path = find_resume_checkpoint(OUTPUT_DIR, EXP_TAG)
            if resume_path is None:
                log_main(rank, "  ⚠️ RESUME=auto 但 OUTPUT_DIR 下没有可续训的 ckpt, 从头训练")
        elif os.path.exists(RESUME_CHECKPOINT):
            resume_path = RESUME_CHECKPOINT
        else:
            log_main(rank, f"  ⚠️ RESUME='{RESUME_CHECKPOINT}' 文件不存在, 从头训练")

    resume_gstep = 0
    resume_best_psnr, resume_best_ssim = 0.0, 0.0
    resume_best_lpips, resume_best_loss = float('inf'), float('inf')
    if resume_path:
        log_main(rank, f"\n  ★★★ 续训: 从 {resume_path} 恢复")
        (resume_gstep, resume_best_psnr, resume_best_ssim,
         resume_best_lpips, resume_best_loss) = load_checkpoint_for_resume(
            model, optimizer, resume_path, OUTPUT_DIR, EXP_TAG, device, rank)
        log_main(rank, f"      → 从 global_step={resume_gstep} 继续 | "
                       f"best_psnr={resume_best_psnr:.3f} best_loss={resume_best_loss:.4f}")
        log_main(rank, f"      → 只有 PSNR>{resume_best_psnr:.3f} / loss<{resume_best_loss:.4f} 才覆盖对应 best ckpt")

    # ---- xyz_offset: 仅从头训练时强制覆盖; 续训保留 ckpt 里已训练好的值 ----
    if not resume_path:
        old_val = gh.xyz_offset_log_scale.item()
        with torch.no_grad():
            gh.xyz_offset_log_scale.copy_(torch.tensor(XYZ_OFFSET_LOG_SCALE_OVERRIDE, device=device,
                                                       dtype=gh.xyz_offset_log_scale.dtype))
        log_main(rank, f"  ★ xyz_offset_log_scale: {old_val:.4f} → {gh.xyz_offset_log_scale.item():.4f} (exp={math.exp(gh.xyz_offset_log_scale.item()):.3f})")
    else:
        log_main(rank, f"  ★ xyz_offset_log_scale (续训保留): {gh.xyz_offset_log_scale.item():.4f} (exp={gh.xyz_offset_log_scale.exp().item():.3f})")

    if LOW_VRAM_MODE:
        log_main(rank, "  LOW_VRAM_MODE: sanity check skipped because PointHead is disabled")
    else:
        sanity_check_xyz_base(model, scenes[0], device, dtype, rank)
    barrier()

    model.gaussian_head = DDP(
        model.gaussian_head,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    if not FREEZE_DPT_FEATURE_HEAD:
        model.dpt_feature_head = DDP(
            model.dpt_feature_head,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    train_model = model
    log_main(rank, f"  ✓ PyTorch DDP initialized for trainable heads on {world_size} GPUs")

    # ---- Scheduler: cosine 退火到 HARD_STOP; 续训时快进到 resume_gstep ----
    sched_total = HARD_STOP_LOCAL
    warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=WARMUP_FACTOR, end_factor=1.0, total_iters=WARMUP_STEPS)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, sched_total - WARMUP_STEPS), eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_STEPS])
    if resume_gstep > 0:
        for _ in range(min(resume_gstep, sched_total)):
            scheduler.step()
        log_main(rank, f"  LR scheduler 已快进到 step {resume_gstep} (lr={optimizer.param_groups[0]['lr']:.2e})")
    log_main(rank, f"  LR 退火地平线 (local steps): {sched_total:,}")

    # ---- 训练循环 ----
    log_main(rank, "\n[训练循环] (纯 geo_init, 零 GT, 按需加载)...")
    log_main(rank, "-" * 70)
    current_dedup = DEDUP_METHOD
    n_mask_slow = 0; MASK_SLOW_THRESHOLD = 5
    # ★ best 指标 / global_step 从续训值起步 (从头训练时即默认值)
    best_loss = resume_best_loss; best_psnr = resume_best_psnr
    best_ssim = resume_best_ssim; best_lpips = resume_best_lpips
    train_start = time.time(); step_times = []
    no_improve = 0; early_stopped = False; n_local_skip = 0
    n_optimizer_updates = 0
    global_step = resume_gstep
    # ★ 续训定位: 跳过已训练过的 epoch / epoch 内已训练的场景 (每 epoch shuffle 由 epoch 决定, 可复现)
    steps_per_epoch = max(1, math.ceil(len(scenes) / world_size))
    start_epoch = resume_gstep // steps_per_epoch
    skip_in_epoch = resume_gstep % steps_per_epoch
    if resume_gstep > 0:
        log_main(rank, f"  ★ 续训定位: 从 epoch {start_epoch+1}/{NUM_EPOCHS} 的第 {skip_in_epoch} 个场景接着跑\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        if early_stopped:
            break
        g = random.Random(epoch * 1000 + 42)
        order = list(range(len(scenes))); g.shuffle(order)
        padding_size = (-len(order)) % world_size
        if padding_size:
            order.extend(order[:padding_size])
        my_order = order[rank::world_size]
        start_idx = skip_in_epoch if epoch == start_epoch else 0

        for data_idx in my_order[start_idx:]:
            if early_stopped:
                break
            if global_step >= HARD_STOP_LOCAL:
                log_main(rank, f"\n  ★★★ Hard stop (step {global_step} >= {HARD_STOP_LOCAL})")
                early_stopped = True; break

            random.seed(global_step * 7919 + 31 + rank)
            step_t0 = time.time()
            local_valid = True
            loss = None; avg_r_value = 0.0; n_render_ok = 0; S_sel = 0; H = W = 0; S_total = 0
            bce_val = ent_val = bal_val = 0.0
            mask_ratio = frame_std = 0.0; surf_val = mid_val = fog_val = 0.0
            opa_pf = [0.0, 0.0, 0.0, 0.0]; mask_el = 0.0; cd_val = 0.0
            geo_scl_val = geo_opa_val = 0.0; selected = []

            try:
                scene_imgs = scenes[data_idx]
                S_total = len(scene_imgs)
                images_input, gt_images_sel, selected, H, W = load_selected_views(
                    scene_imgs, TRAIN_VIEWS_PER_STEP, device)
                S_sel = len(selected)
                if LOG_ALL_RANK_SHAPES:
                    log_all(
                        rank,
                        f"[rank {rank}] step{global_step} input S={S_sel} H={H} W={W} "
                        f"scene={Path(scene_imgs[0]).parent.name}",
                    )

                train_model.train()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = train_model(images_input)

                if 'color_delta_norm' in predictions.get('gaussians', {}):
                    cd_raw = predictions['gaussians']['color_delta_norm']
                    cd_val = float(cd_raw) if not torch.is_tensor(cd_raw) else float(cd_raw.item())

                with torch.no_grad():
                    extr_pred, intr_pred = pose_encoding_to_extri_intri(predictions['pose_enc'], image_size_hw=(H, W))
                    extr_pred = extr_pred.float(); intr_pred = intr_pred.float()

                # phase 退火 geo 强度 + render 权重
                if global_step < PHASE2_LOCAL:
                    geo_scale, w_render = 1.0, W_RENDER
                elif global_step < PHASE3_LOCAL:
                    geo_scale, w_render = 0.3, W_RENDER * 2.0
                else:
                    geo_scale, w_render = 0.05, W_RENDER * 2.0

                with torch.no_grad():
                    xyz_base_pred = unproject_depth_to_world_torch(predictions['depth'].float(), extr_pred, intr_pred)
                    depth_conf_pred = predictions['depth_conf'].float()

                # ★ 坏场景防护: forward 在退化场景可能产出 NaN/Inf。这些值若流进 dedup
                #   mask / BCE / render 会触发不可恢复的 device-side assert。这里在 CUDA
                #   报错之前主动检测并 raise, 走下面已有的 skip 兜底 (不污染 CUDA context)。
                #   单次 .item() 同步, 开销可忽略。
                with torch.no_grad():
                    _g = predictions['gaussians']
                    _ok = (torch.isfinite(xyz_base_pred).all()
                           & torch.isfinite(depth_conf_pred).all()
                           & torch.isfinite(_g['xyz']).all()
                           & torch.isfinite(_g['scale']).all()
                           & torch.isfinite(_g['opacity']).all())
                if not bool(_ok.item()):
                    raise RuntimeError("nonfinite_pred")

                # geo-init 冷启动锚
                l_geo = torch.tensor(0.0, device=device)
                if geo_scale > 0:
                    l_geo, geo_scl_val, geo_opa_val = geo_init_param_loss(
                        predictions['gaussians'], xyz_base_pred, S_sel, H, W,
                        GEO_SCALE_K, GEO_OPA_CONST, W_GEO_SCALE, W_GEO_OPA, geo_scale)

                log_timing = (global_step < DEDUP_TIMING_LOG_EARLY or (global_step + 1) % DEDUP_TIMING_LOG_INTERVAL == 0)
                dedup_mask, mask_el, _ = safe_compute_dedup_mask(
                    method=current_dedup, xyz_base=xyz_base_pred, depth_conf=depth_conf_pred,
                    extr=extr_pred, intr=intr_pred, H=H, W=W, global_step=global_step,
                    log_timing=log_timing, rank=rank)
                if mask_el > DEDUP_TIMEOUT_SEC:
                    n_mask_slow += 1
                    if n_mask_slow >= MASK_SLOW_THRESHOLD and current_dedup != 'none':
                        log_all(rank, f"  ⚠️ mask 累计超时, 切到 'none'"); current_dedup = 'none'

                pred_opa = predictions['gaussians']['opacity'].float()
                l_bce = opacity_bce_loss(pred_opa, dedup_mask)
                l_ent = opacity_entropy_loss(pred_opa)
                l_bal = opacity_balance_loss(pred_opa)
                bce_val, ent_val, bal_val = float(l_bce.item()), float(l_ent.item()), float(l_bal.item())

                with torch.no_grad():
                    mask_ratio = float(dedup_mask.mean().item())
                    pf = dedup_mask[0].mean(dim=(1, 2)).cpu().numpy(); frame_std = float(pf.std())
                    surf_val = float((pred_opa > 0.5).float().mean().item())
                    mid_val = float(((pred_opa > 0.1) & (pred_opa < 0.5)).float().mean().item())
                    fog_val = float((pred_opa < 0.1).float().mean().item())
                    opa_pf = pred_opa[0].mean(dim=(1, 2)).cpu().numpy().tolist()[:4]
                    while len(opa_pf) < 4:
                        opa_pf.append(0.0)

                pred_scene = {k: v[0, :S_sel].reshape(-1, v.shape[-1]).float()
                              for k, v in predictions['gaussians'].items()
                              if isinstance(v, torch.Tensor) and v.dim() >= 4}

                r_losses = []; oom_fb = False
                for si in range(S_sel):
                    try:
                        rl, _ = compute_render_loss(pred_scene, gt_images_sel[si].float(),
                                                    extr_pred[0, si], intr_pred[0, si], H, W)
                        r_losses.append(rl)
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            torch.cuda.empty_cache(); oom_fb = True; break
                        raise
                if oom_fb:
                    r_losses = []
                    ti = random.randint(0, S_sel - 1)
                    try:
                        rl, _ = compute_render_loss(pred_scene, gt_images_sel[ti].float(),
                                                    extr_pred[0, ti], intr_pred[0, ti], H, W)
                        r_losses.append(rl)
                    except RuntimeError:
                        torch.cuda.empty_cache()

                n_render_ok = len(r_losses)
                avg_r = (sum(r_losses) / len(r_losses)) if r_losses else torch.tensor(0.0, device=device)
                avg_r_value = float(avg_r.item())
                del pred_scene
                if n_render_ok == 0:
                    log_all(rank, f"  ⚠️ render 全失败 step{global_step} S={S_sel} H={H} W={W}")
                    raise RuntimeError("render_all_failed")

                loss = (l_geo + w_render * avg_r
                        + W_OPA_BCE * l_bce + W_OPA_ENTROPY * l_ent + W_OPA_BALANCE * l_bal)

            except RuntimeError as e:
                local_valid = False
                es = str(e).lower()
                bad_scene = scenes[data_idx][0] if 0 <= data_idx < len(scenes) else '?'
                bad_name = Path(bad_scene).parent.name if bad_scene != '?' else '?'
                if 'device-side assert' in es or 'cuda error' in es or 'illegal memory access' in es:
                    # ★ 不可恢复: CUDA context 已损坏, 连 empty_cache 都会再抛同样的错。
                    #   打印肇事场景后干净退出, 不再做任何 CUDA 调用。
                    log_all(rank, "\n" + "=" * 70)
                    log_all(rank, f"  ✗✗✗ 不可恢复的 CUDA 错误 @ step{global_step} — 进程退出")
                    log_all(rank, f"      肇事场景: {bad_scene}")
                    log_all(rank, f"      场景目录: {Path(bad_scene).parent}  (S_total={S_total} H={H} W={W})")
                    log_all(rank, f"      原始错误: {e}")
                    log_all(rank, "      建议: ① 把该场景目录从 TRAIN_DIR 移走再重训; 或")
                    log_all(rank, "            ② 等 depth_conf scan 跑完, 加筛选挡掉这类坏场景; 或")
                    log_all(rank, "            ③ 用 CUDA_LAUNCH_BLOCKING=1 重跑定位真正的 kernel。")
                    log_all(rank, "=" * 70)
                    sys.stdout.flush()
                    os._exit(1)          # 硬退出, 不触碰已损坏的 CUDA/NCCL
                elif 'out of memory' in es:
                    log_all(rank, f"  ⚠️ OOM step{global_step} S={S_sel} H={H} W={W} 场景={bad_name}, skip")
                    n_local_skip += 1
                    torch.cuda.empty_cache()
                elif 'nonfinite_pred' in es:
                    log_all(rank, f"  ⚠️ step{global_step} 跳过 (forward 出 NaN/Inf) 场景={bad_name}")
                    n_local_skip += 1
                    torch.cuda.empty_cache()
                elif 'render_all_failed' in es:
                    log_all(rank, f"  ⚠️ step{global_step} 跳过 (render 全失败) 场景={bad_name}")
                    n_local_skip += 1
                    torch.cuda.empty_cache()
                else:
                    log_all(rank, f"  ⚠️ step{global_step} 失败 场景={bad_name}: {e}")
                    n_local_skip += 1
                    torch.cuda.empty_cache()

            valid_count = torch.tensor([int(local_valid)], device=device, dtype=torch.int)
            dist.all_reduce(valid_count, op=dist.ReduceOp.SUM)
            n_valid = int(valid_count.item())
            step_updated = n_valid == world_size
            if step_updated:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step(); scheduler.step()
                n_optimizer_updates += 1
            else:
                optimizer.zero_grad(set_to_none=True)
            step_times.append(time.time() - step_t0)

            if rank == 0 and (global_step + 1) % LOG_LOCAL == 0:
                elapsed = time.time() - train_start
                avg_step = sum(step_times[-LOG_LOCAL:]) / len(step_times[-LOG_LOCAL:])
                remaining = avg_step * (HARD_STOP_LOCAL - global_step - 1)
                cur_lr = optimizer.param_groups[0]['lr']
                phase = "P1" if global_step < PHASE2_LOCAL else ("P2" if global_step < PHASE3_LOCAL else "P3")
                in_wm = global_step < WARMUP_STEPS
                wm = f"WM{global_step}/{WARMUP_STEPS} " if in_wm else ""
                xyz_off = gh.xyz_offset_log_scale.exp().item()
                mem = f" peak={torch.cuda.max_memory_allocated()/1024/1024:.0f}MB" if torch.cuda.is_available() else ""
                if step_updated:
                    pf_str = ','.join(f'{x:.2f}' for x in opa_pf[:S_sel])
                    sel_str = ','.join(str(x) for x in selected[:5])
                    log_main(rank,
                             f"  [nogt|{phase}|{wm}] step{global_step:7d} ep{epoch+1}/{NUM_EPOCHS} | L={loss.item():.3f} "
                             f"rnd={avg_r_value:.3f}({n_render_ok}/{S_sel}v) ★geo=s{geo_scl_val:.2f}/o{geo_opa_val:.2f} "
                             f"bce={bce_val:.3f} ent={ent_val:.3f} bal={bal_val:.4f}\n"
                             f"               mask={mask_ratio:.2f}({current_dedup},{mask_el:.2f}s) fstd={frame_std:.3f} "
                             f"surf={surf_val:.3f} mid={mid_val:.3f} fog={fog_val:.3f} opa_pf=[{pf_str}] "
                             f"xyz_off={xyz_off:.3f} ★cd={cd_val:.4f} sel=[{sel_str}] "
                             f"valid={n_valid}/{world_size} lskip={n_local_skip} S={S_sel}/{S_total} H={H} W={W} "
                             f"lr={cur_lr:.1e}{mem} | {format_duration(elapsed)}/{format_duration(remaining)}")
                else:
                    log_main(rank, f"  [nogt|{phase}|{wm}] step{global_step:7d} ★rank0 SKIPPED★ "
                                   f"valid={n_valid}/{world_size} lskip={n_local_skip} | {format_duration(elapsed)}/{format_duration(remaining)}")

            if step_updated and loss is not None:
                cur_loss = loss.item()
                is_best_loss = cur_loss < best_loss
                if is_best_loss:
                    best_loss = cur_loss
            else:
                is_best_loss = False; cur_loss = best_loss

            is_eval = (global_step + 1) % EVAL_LOCAL == 0
            is_save = (global_step + 1) % SAVE_LOCAL == 0

            if is_eval:
                if rank == 0 and eval_test_data is not None:
                    p, s, lp, summary = evaluate_on_test_set(model, eval_test_data, device, dtype, global_step, RENDER_DIR, save_renders=3)
                    is_best = math.isfinite(p) and p < PSNR_SANE_MAX and p > best_psnr
                    if is_best:
                        best_psnr, best_ssim = p, s
                        if lp >= 0:
                            best_lpips = lp
                        no_improve = 0
                        save_checkpoint(model, optimizer, global_step, epoch, cur_loss,
                                        os.path.join(OUTPUT_DIR, f"gaussian_head_best_test_{EXP_TAG}.pth"),
                                        metrics={'psnr': p, 'ssim': s, 'lpips': lp},
                                        best_state=_best_state(best_psnr, best_ssim, best_lpips, best_loss))
                    else:
                        no_improve += 1
                    lps = f" LPIPS={lp:.4f}" if lp >= 0 else ""
                    blps = f" LPIPS={best_lpips:.4f}" if best_lpips < float('inf') else ""
                    log_main(rank, f"    ★ [nogt] Eval ({len(eval_test_data)}场景): PSNR={p:.2f}dB SSIM={s:.4f}{lps} "
                                   f"(best: {best_psnr:.2f}dB{blps}) [no_imp={no_improve}/{TEST_PATIENCE}] {summary}")
                    if is_best_loss:
                        save_checkpoint(model, optimizer, global_step, epoch, cur_loss,
                                        os.path.join(OUTPUT_DIR, f"gaussian_head_best_loss_{EXP_TAG}.pth"),
                                        best_state=_best_state(best_psnr, best_ssim, best_lpips, best_loss))
                    if is_save:
                        save_checkpoint(model, optimizer, global_step, epoch, cur_loss,
                                        os.path.join(OUTPUT_DIR, f"gaussian_head_step{global_step+1}_{EXP_TAG}.pth"),
                                        best_state=_best_state(best_psnr, best_ssim, best_lpips, best_loss))
                stop = torch.zeros(1, device=device, dtype=torch.int)
                if rank == 0 and global_step >= MIN_TRAIN_LOCAL and no_improve >= TEST_PATIENCE:
                    stop[0] = 1
                if dist.is_initialized():
                    dist.broadcast(stop, src=0)
                if stop.item() == 1:
                    early_stopped = True
                    log_main(rank, f"\n  ★★★ 早停 (local step {global_step})"); break
                barrier(); model.train()
            elif is_save or is_best_loss:
                if rank == 0:
                    if is_best_loss:
                        save_checkpoint(model, optimizer, global_step, epoch, cur_loss,
                                        os.path.join(OUTPUT_DIR, f"gaussian_head_best_loss_{EXP_TAG}.pth"),
                                        best_state=_best_state(best_psnr, best_ssim, best_lpips, best_loss))
                    if is_save:
                        save_checkpoint(model, optimizer, global_step, epoch, cur_loss,
                                        os.path.join(OUTPUT_DIR, f"gaussian_head_step{global_step+1}_{EXP_TAG}.pth"),
                                        best_state=_best_state(best_psnr, best_ssim, best_lpips, best_loss))
                if is_save:
                    barrier()

            global_step += 1
            # ★ 显式释放本步所有大张量再清缓存。empty_cache 只还"未被引用"的碎片,
            #   被局部变量引用的张量不会还。之前只 del predictions/loss, 导致
            #   pred_scene / xyz_base_pred / dedup_mask 等积压到下一步, 一旦某步 OOM
            #   就连环传染。这里逐个字面量 del (未定义会抛 NameError, 忽略即可;
            #   注意: locals() 删除 / exec('del x') 在函数作用域内都无效, 必须字面量 del)。
            try: del predictions
            except Exception: pass
            try: del pred_scene
            except Exception: pass
            try: del xyz_base_pred
            except Exception: pass
            try: del depth_conf_pred
            except Exception: pass
            try: del dedup_mask
            except Exception: pass
            try: del pred_opa
            except Exception: pass
            try: del images_input
            except Exception: pass
            try: del gt_images_sel
            except Exception: pass
            try: del loss
            except Exception: pass
            try: del avg_r
            except Exception: pass
            try: del l_geo
            except Exception: pass
            try: del _g
            except Exception: pass
            try: del _ok
            except Exception: pass
            torch.cuda.empty_cache()

        if not early_stopped:
            barrier()
            log_main(rank, f"\n  === [nogt] Epoch {epoch+1}/{NUM_EPOCHS} 完成 | local step {global_step} | {format_duration(time.time()-train_start)} ===\n")

    update_min = torch.tensor(n_optimizer_updates, device=device, dtype=torch.long)
    update_max = update_min.clone()
    dist.all_reduce(update_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(update_max, op=dist.ReduceOp.MAX)
    if update_min.item() != update_max.item():
        raise RuntimeError(
            f"DDP rank update-count mismatch: min={update_min.item()} "
            f"max={update_max.item()}"
        )
    log_main(rank, f"  ✓ DDP rank update counts agree: {update_min.item()}")

    # ---- 最终评估 ----
    barrier()
    if rank == 0 and all_test_data:
        log_main(rank, f"\n  ★ [nogt] 最终完整评估: {len(all_test_data)} 个测试场景...")
        fp, fs, flp, _ = evaluate_on_test_set(model, all_test_data, device, dtype, global_step, RENDER_DIR,
                                              save_renders=min(len(all_test_data), 20))
        log_main(rank, "\n" + "=" * 70)
        log_main(rank, "  [nogt] 训练完成! (完全去 GT + 无数据预处理)")
        log_main(rank, "=" * 70)
        log_main(rank, f"  ---------- 最终 (全 {len(all_test_data)} 场景) ----------")
        log_main(rank, f"  PSNR : {fp:.4f} dB | SSIM : {fs:.6f} | LPIPS: {flp:.6f}")
        log_main(rank, f"  ---------- 训练中最佳 ({len(eval_test_data)} 场景) ----------")
        log_main(rank, f"  PSNR : {best_psnr:.4f} dB | SSIM : {best_ssim:.6f}"
                       + (f" | LPIPS: {best_lpips:.6f}" if best_lpips < float('inf') else ""))
        log_main(rank, f"  训练耗时 : {format_duration(time.time()-train_start)}")
        log_main(rank, f"  DDP optimizer updates: {n_optimizer_updates}")
        log_main(rank, f"  最佳 ckpt: {os.path.join(OUTPUT_DIR, f'gaussian_head_best_test_{EXP_TAG}.pth')}")
        log_main(rank, "  对比基线 : v28 (geo_init + gt_opa_std 筛选) 全集 23.31 / 最佳子集 24.23 dB")
        log_main(rank, "=" * 70)
    elif rank == 0:
        log_main(rank, "\n" + "=" * 70)
        log_main(rank, "  [nogt] DDP smoke/training run complete (test evaluation skipped)")
        log_main(rank, f"  DDP optimizer updates: {n_optimizer_updates}")
        log_main(rank, f"  Elapsed: {format_duration(time.time()-train_start)}")
        log_main(rank, "=" * 70)

    barrier()
    cleanup_ddp()


if __name__ == "__main__":
    main()
