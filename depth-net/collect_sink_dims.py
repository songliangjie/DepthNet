#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 VAR 所需的 sink 维度 (D_sink) 脚本

功能：
    1. 遍历测试集 (IR/VIS) 前向推理，提取 IR 瓶颈特征和 Depth 分支多尺度特征
    2. 对每个 token 做 RMS 归一化，统计各通道激活均值
    3. 输出每个尺度 top-K sink 维度索引
    4. 可选：第二遍计算 phi(x) 分布，方便选择阈值 tau

运行示例：
    python collect_sink_dims.py --checkpoint ./model/checkpoint_epoch_143.pth --topk 8 \
        --max-samples 200 --batch-size 2 --compute-phi
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from net import Net
from moudle import fusiondata
import utils

EPS = 1e-6
# 需要统计的特征名称
FEATURE_KEYS = ("ir5", "dep2", "dep3", "dep4", "dep5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect sink dims statistics")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=utils.MODEL_PATH,
        help="模型 checkpoint 路径 (state_dict)",
    )
    parser.add_argument(
        "--ir-dir",
        type=str,
        default=utils.TEST_IR_DIR,
        help="IR 图像目录",
    )
    parser.add_argument(
        "--vi-dir",
        type=str,
        default=utils.TEST_VI_DIR,
        help="VIS 图像目录",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="统计时的 batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最多统计多少张样本 (0 表示使用全部)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="输出的 sink 维度数量",
    )
    parser.add_argument(
        "--compute-phi",
        action="store_true",
        help="在得到 D_sink 后再跑一遍统计 phi 分布",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="可选：把统计结果保存到 json 文件",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备 (cuda / cpu)",
    )
    return parser.parse_args()


def load_checkpoint(model: Net, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = None
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    elif isinstance(ckpt, torch.nn.Module):
        state_dict = ckpt.state_dict()

    if state_dict is None:
        raise RuntimeError(f"无法解析 checkpoint：{ckpt_path}")

    model.load_state_dict(state_dict, strict=True)


def rms_normalize(tokens: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(tokens ** 2, dim=1, keepdim=True) + EPS)
    return tokens / rms


def flatten_tokens(feat: torch.Tensor) -> torch.Tensor:
    # [B, C, H, W] -> [B*H*W, C]
    b, c, h, w = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(b * h * w, c)


def extract_features(dual_unet, ir: torch.Tensor, vis: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    手动展开 DualBranchUNet，拿到需要的中间特征
    """
    feats: Dict[str, torch.Tensor] = {}

    ir1 = dual_unet.ir_inc(ir)
    ir2 = dual_unet.ir_down1(ir1)
    ir3 = dual_unet.ir_down2(ir2)
    ir4 = dual_unet.ir_down3(ir3)
    ir5 = dual_unet.ir_down4(ir4)
    feats["ir5"] = ir5

    depth_feats = dual_unet.depth_branch.forward_encoder(vis, ir)
    if len(depth_feats) != 5:
        raise RuntimeError("Depth branch forward_encoder 应返回 5 个尺度特征")

    feats["dep2"] = depth_feats[1]
    feats["dep3"] = depth_feats[2]
    feats["dep4"] = depth_feats[3]
    feats["dep5"] = depth_feats[4]
    return feats


def iterate_batches(
    loader: DataLoader,
    max_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    带样本上限的 batch 迭代器
    """
    processed = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            ir = batch[0]
            vis = batch[1]
        else:
            raise RuntimeError("Dataset 输出格式异常，期望 (IR, VIS)")

        if max_samples > 0:
            if processed >= max_samples:
                break
            remain = max_samples - processed
            if ir.size(0) > remain:
                ir = ir[:remain]
                vis = vis[:remain]
        processed += ir.size(0)
        yield ir, vis


def accumulate_channel_stats(
    stats: Dict[str, Dict[str, torch.Tensor]],
    feats: Dict[str, torch.Tensor],
):
    """
    更新每个尺度的通道绝对值均值统计
    """
    for name, tensor in feats.items():
        tokens = flatten_tokens(tensor)
        tokens = rms_normalize(tokens)
        channel_sum = tokens.abs().sum(dim=0).double().cpu()
        count = tokens.shape[0]

        if name not in stats:
            stats[name] = {
                "sum": torch.zeros_like(channel_sum),
                "count": 0,
            }

        stats[name]["sum"] += channel_sum
        stats[name]["count"] += count


def compute_topk(stats: Dict[str, Dict[str, torch.Tensor]], topk: int) -> Dict[str, List[int]]:
    sink_dims: Dict[str, List[int]] = {}
    for name in FEATURE_KEYS:
        if name not in stats:
            continue
        channel_mean = stats[name]["sum"] / max(1, stats[name]["count"])
        k = min(topk, channel_mean.numel())
        values, indices = torch.topk(channel_mean, k)
        sink_dims[name] = indices.tolist()
        print(f"[{name}] top-{k} sink dims: {indices.tolist()}")
        print(f"          mean activations: {values.tolist()}")
    return sink_dims


def accumulate_phi_stats(
    phi_stats: Dict[str, Dict[str, float]],
    feats: Dict[str, torch.Tensor],
    sink_dims: Dict[str, List[int]],
):
    for name, tensor in feats.items():
        dims = sink_dims.get(name)
        if not dims:
            continue
        tokens = flatten_tokens(tensor)
        tokens = rms_normalize(tokens)
        sel = tokens[:, dims]
        phi = torch.max(sel, dim=1).values

        if name not in phi_stats:
            phi_stats[name] = {"sum": 0.0, "max": 0.0, "count": 0}

        phi_stats[name]["sum"] += float(phi.sum().item())
        phi_stats[name]["max"] = max(phi_stats[name]["max"], float(phi.max().item()))
        phi_stats[name]["count"] += phi.numel()


def summarize_phi(phi_stats: Dict[str, Dict[str, float]]) -> None:
    for name in FEATURE_KEYS:
        if name not in phi_stats or phi_stats[name]["count"] == 0:
            continue
        mean_phi = phi_stats[name]["sum"] / phi_stats[name]["count"]
        max_phi = phi_stats[name]["max"]
        print(f"[{name}] phi -> mean: {mean_phi:.4f}, max: {max_phi:.4f}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = fusiondata(
        ir_dir=args.ir_dir,
        vi_dir=args.vi_dir,
        depth_dir=None,
        train=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Net().to(device)
    model.eval()
    load_checkpoint(model, args.checkpoint, device)
    torch.set_grad_enabled(False)

    channel_stats: Dict[str, Dict[str, torch.Tensor]] = {}
    total_samples = 0

    print("===> [Pass 1] 统计通道均值，用于挑选 D_sink")
    for ir, vis in tqdm(iterate_batches(loader, args.max_samples), unit="batch"):
        ir = (ir / 255.0).to(device)
        vis = (vis / 255.0).to(device)
        feats = extract_features(model.dual_unet, ir, vis)
        accumulate_channel_stats(channel_stats, feats)
        total_samples += ir.size(0)

    print(f"样本数量：{total_samples}")
    sink_dims = compute_topk(channel_stats, args.topk)

    if args.compute_phi:
        print("\n===> [Pass 2] 计算 phi 分布（用于验证 tau）")
        phi_stats: Dict[str, Dict[str, float]] = {}
        for ir, vis in tqdm(iterate_batches(loader, args.max_samples), unit="batch"):
            ir = (ir / 255.0).to(device)
            vis = (vis / 255.0).to(device)
            feats = extract_features(model.dual_unet, ir, vis)
            accumulate_phi_stats(phi_stats, feats, sink_dims)
        summarize_phi(phi_stats)

    if args.save_json:
        payload = {
            "checkpoint": args.checkpoint,
            "topk": args.topk,
            "sink_dims": sink_dims,
            "samples": total_samples,
        }
        Path(args.save_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n统计结果已写入 {args.save_json}")


if __name__ == "__main__":
    main()









