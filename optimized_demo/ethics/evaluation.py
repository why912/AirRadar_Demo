import torch
import numpy as np
from typing import List, Tuple

"""
伦理评估工具（占位实现）：
- 区域 / 节点密度分层公平性评估
- 简单能耗估算 (样本数 * epoch * 系数)
- 频域能量 / 注意力可解释接口占位
"""


def estimate_energy(num_samples: int, epochs: int, device: str) -> float:
    device_factor = 1.0 if device == "cpu" else 0.6  # 假设 GPU 更高能效（占位）
    return num_samples * epochs * device_factor / 1e6  # 兆样本系数占位


def slice_fairness(preds: torch.Tensor, labels: torch.Tensor, split: str, meta: dict = None):
    """按密度 / 区域切片的公平性评估占位。
    meta 中包含: node_density: [N] 或 region_id: [N]
    输出各组 MAE 与整体偏差。
    """
    if meta is None:
        print(f"[Fairness:{split}] meta 未提供，跳过公平性评估。")
        return
    groups = None
    if "region_id" in meta:
        groups = meta["region_id"]
        name = "region_id"
    elif "node_density" in meta:
        groups = meta["node_density"]
        name = "node_density"
    else:
        print(f"[Fairness:{split}] 未找到 region_id/node_density，跳过。")
        return
    groups = torch.tensor(groups)
    unique = torch.unique(groups)
    results = []
    for g in unique:
        mask = groups == g
        g_preds = preds[mask]
        g_labels = labels[mask]
        if g_labels.numel() == 0:
            continue
        mae = torch.abs(g_preds - g_labels).mean().item()
        results.append((g.item(), mae))
    if not results:
        print(f"[Fairness:{split}] 无有效分组。")
        return
    maes = [r[1] for r in results]
    disparity = max(maes) - min(maes)
    print(f"[Fairness:{split}] 分组: {name}, 组数={len(results)}, 最大差异={disparity:.4f}")
    for gid, m in results:
        print(f"  组 {gid}: MAE={m:.4f}")


def explain_frequency(x: torch.Tensor) -> dict:
    """频域能量分布占位：返回低/中/高频能量比例。"""
    if x.dim() != 3:
        return {"error": "shape must be [B,N,C]"}
    ffted = torch.fft.rfft(x, dim=1)
    energy = torch.abs(ffted) ** 2
    total = energy.sum().item() + 1e-7
    n = energy.shape[1]
    low = energy[:, : n // 4].sum().item() / total
    mid = energy[:, n // 4 : n // 2].sum().item() / total
    high = energy[:, n // 2 :].sum().item() / total
    return {"low": low, "mid": mid, "high": high}


def ethics_summary(num_samples: int, epochs: int, device: str, preds: torch.Tensor, labels: torch.Tensor) -> dict:
    summary = {}
    summary["energy_estimate"] = estimate_energy(num_samples, epochs, device)
    summary["mae"] = torch.abs(preds - labels).mean().item()
    summary["frequency_profile"] = explain_frequency(preds.unsqueeze(0).repeat(1, 1, 1))
    return summary
