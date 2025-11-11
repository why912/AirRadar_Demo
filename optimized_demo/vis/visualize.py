import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import matplotlib.tri as mtri


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_training_curves(history: Dict, out_dir: str):
    ensure_dir(out_dir)
    epochs = history.get('epoch', list(range(len(history.get('train_loss', [])))))
    # Loss curve
    plt.figure(figsize=(6,4), dpi=150)
    plt.plot(epochs, history.get('train_loss', []), label='TrainLoss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'curve_train_loss.png'))
    plt.close()
    # Val curves
    plt.figure(figsize=(6,4), dpi=150)
    plt.plot(epochs, history.get('val_mae', []), label='Val MAE')
    plt.plot(epochs, history.get('val_rmse', []), label='Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Validation Metrics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'curve_val_metrics.png'))
    plt.close()


def plot_pred_vs_true_series(
    series_pred: np.ndarray,
    series_true: np.ndarray,
    out_path: str,
    title: str = 'Pred vs True (Time Series)',
    normalize: bool = False,
    match_amplitude: bool = False,
    smooth_win: Optional[int] = None,
):
    """Plot predicted vs true time series for a single node over multiple timesteps.

    Options:
    - normalize: z-score both curves以对齐均值方差，用于纯形态对比；
    - match_amplitude: 将预测曲线仿射缩放到与真实曲线的均值/方差一致（更直观对齐峰谷幅度，避免误判）；
    - smooth_win: 移动平均平滑窗口（奇数），用于降低噪声、对齐高频抖动。
    """
    pred = series_pred.astype(float).copy()
    true = series_true.astype(float).copy()

    def _smooth(x: np.ndarray, k: int) -> np.ndarray:
        if k is None or k <= 1:
            return x
        k = int(k)
        if k % 2 == 0:
            k += 1
        kernel = np.ones(k, dtype=float) / k
        return np.convolve(x, kernel, mode='same')

    eps = 1e-8
    if normalize:
        pred = (pred - pred.mean()) / max(pred.std(), eps)
        true = (true - true.mean()) / max(true.std(), eps)
    elif match_amplitude:
        # 将 pred 仿射到与 true 相同的均值/方差
        pm, ps = pred.mean(), pred.std()
        tm, ts = true.mean(), true.std()
        if ps < eps:
            ps = eps
        pred = (pred - pm) / ps * (ts if ts > eps else 1.0) + tm

    # 平滑（最后做）
    if smooth_win and smooth_win > 1:
        pred = _smooth(pred, smooth_win)
        true = _smooth(true, smooth_win)

    T = len(pred)
    x = np.arange(T)
    plt.figure(figsize=(8,4), dpi=150)
    plt.plot(x, true, label='True', linewidth=2)
    plt.plot(x, pred, label='Pred', linewidth=2)
    plt.xlabel('Sample Index (time order)')
    plt.ylabel('Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_spatial_scatter(pos: np.ndarray, values: np.ndarray, out_path: str, title: str = 'Spatial Map', vmin: Optional[float] = None, vmax: Optional[float] = None):
    """Scatter plot over lon/lat positions colored by values."""
    plt.figure(figsize=(5.5,5), dpi=150)
    sc = plt.scatter(pos[:,0], pos[:,1], c=values, cmap='viridis', s=35, vmin=vmin, vmax=vmax, edgecolors='k', linewidths=0.3)
    plt.colorbar(sc, shrink=0.8, label='Value')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_spatial_heatmap(pos: np.ndarray, values: np.ndarray, out_path: str, title: str = 'Spatial Heatmap', levels: int = 32, cmap: str = 'turbo', vmin: Optional[float] = None, vmax: Optional[float] = None):
    """Filled contour heatmap based on triangulation (no SciPy dependency)."""
    x = pos[:, 0]
    y = pos[:, 1]
    triang = mtri.Triangulation(x, y)
    plt.figure(figsize=(6,5.5), dpi=150)
    cs = plt.tricontourf(triang, values, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.tricontour(triang, values, levels=levels//4, colors='k', linewidths=0.2, alpha=0.4)
    plt.scatter(x, y, c=values, cmap=cmap, edgecolors='k', linewidths=0.2, s=15)
    cbar = plt.colorbar(cs, shrink=0.85, label='Value')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def prepare_time_series_from_loader(preds: np.ndarray, labels: np.ndarray, masks: np.ndarray, node_idx: int, max_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate per-sample predictions for a fixed node across the dataset order."""
    # preds/labels: [S, N, 1], masks: [S, N]
    preds_node = preds[:, node_idx, 0]
    labels_node = labels[:, node_idx, 0]
    valid = masks[:, node_idx] > 0
    preds_node = preds_node[valid]
    labels_node = labels_node[valid]
    if len(preds_node) > max_points:
        preds_node = preds_node[:max_points]
        labels_node = labels_node[:max_points]
    return preds_node, labels_node
