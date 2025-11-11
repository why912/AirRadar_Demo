import torch
import numpy as np
from torch import nn, Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from typing import Optional, List, Dict

from src.utils.metrics import masked_mae, masked_rmse, masked_mse


class TrainerPlus:
    """
    改进版 Trainer：
    - 统一 forward(inputs, history, mask_nodes, pred_attr, g=None) 接口
    - 动态设备放置，不强制 CUDA
    - 伦理钩子：在每个 eval 后可注入自定义分析（fairness_callbacks）
    - 简化早停逻辑
    """

    def __init__(
        self,
        model: nn.Module,
        data: Dict[str, any],
        mask_nodes: Dict[str, List[int]],
        base_lr: float = 5e-4,
        steps: List[int] = None,
        lr_decay_ratio: float = 0.5,
        device: Optional[torch.device] = None,
        max_epochs: int = 5,
        patience: int = 3,
        clip_grad_value: Optional[float] = 5.0,
        pred_attr: str = "PM25",
        fairness_callbacks: Optional[List] = None,
    ):
        self.model = model
        self.data = data
        self.mask_nodes = mask_nodes
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        # 确保学习率为 float 类型（防止从 YAML 读取成字符串）
        lr = float(base_lr)
        self.optimizer = Adam(self.model.parameters(), lr)
        self.lr_scheduler = (
            MultiStepLR(self.optimizer, steps, gamma=lr_decay_ratio) if steps and lr_decay_ratio != 1 else None
        )
        self.max_epochs = max_epochs
        self.patience = patience
        self.clip_grad_value = clip_grad_value
        self.loss_fn = masked_mae
        self.pred_attr = pred_attr
        self.fairness_callbacks = fairness_callbacks or []
        self.best_val = np.inf
        self.no_improve = 0

    def _inv_transform(self, tensors: Tensor):
        # 简化：假设 scalers[0] 对 PM25
        scalers = self.data["scalers"]
        target_idx = 0
        tensors[..., 0] = scalers[target_idx].inverse_transform(tensors[..., 0])
        return tensors

    def train_batch(self, X: Tensor, Y: Tensor, History: Tensor):
        self.optimizer.zero_grad()
        pred, mask = self.model(X, History, None, self.pred_attr)
        # 取目标通道
        if self.pred_attr == "PM25":
            label = Y[..., 0].unsqueeze(-1)
        pred = self._inv_transform(pred)
        mask = 1 - mask
        idx = mask.nonzero(as_tuple=True)
        pred = pred[idx]
        label = label[idx]
        valid = label > 0
        pred = pred[valid]
        label = label[valid]
        loss = self.loss_fn(pred, label, 0.0)
        loss.backward()
        if self.clip_grad_value:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_value)
        self.optimizer.step()
        return loss.item()

    def evaluate(self, split="val"):
        preds, labels, masks = [], [], []
        self.model.eval()
        m_nodes = self.mask_nodes[split]
        with torch.no_grad():
            for X, Y, H in self.data[f"{split}_loader"]:
                X, Y, H = X.to(self.device), Y.to(self.device), H.to(self.device)
                pred, mask = self.model(X, H, m_nodes, self.pred_attr)
                if self.pred_attr == "PM25":
                    label = Y[..., 0].unsqueeze(-1)
                pred = self._inv_transform(pred)
                mask = 1 - mask
                preds.append(pred.cpu())
                labels.append(label.cpu())
                masks.append(mask.cpu())
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        masks = torch.cat(masks, dim=0).unsqueeze(-1)
        idx = masks.nonzero(as_tuple=True)
        preds = preds[idx]
        labels = labels[idx]
        valid = labels > 0
        preds = preds[valid]
        labels = labels[valid]
        mae = torch.abs(preds - labels).mean().item()
        rmse = torch.sqrt(((preds - labels) ** 2).mean()).item()
        for cb in self.fairness_callbacks:
            try:
                cb(preds, labels, split)
            except Exception as e:
                print(f"Fairness callback error: {e}")
        return mae, rmse

    def train(self):
        # 初始化历史记录容器（用于可视化）
        if not hasattr(self, 'history'):
            self.history = {'epoch': [], 'train_loss': [], 'val_mae': [], 'val_rmse': []}
        for epoch in range(self.max_epochs):
            self.model.train()
            losses = []
            for X, Y, H in self.data["train_loader"]:
                X, Y, H = X.to(self.device), Y.to(self.device), H.to(self.device)
                losses.append(self.train_batch(X, Y, H))
            val_mae, val_rmse = self.evaluate("val")
            print(
                f"[Epoch {epoch}] TrainLoss={np.mean(losses):.4f} ValMAE={val_mae:.4f} ValRMSE={val_rmse:.4f}"
            )
            # 记录历史
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(float(np.mean(losses)))
            self.history['val_mae'].append(val_mae)
            self.history['val_rmse'].append(val_rmse)
            if val_mae < self.best_val:
                self.best_val = val_mae
                self.no_improve = 0
            else:
                self.no_improve += 1
            if self.no_improve >= self.patience:
                print("Early stopping triggered.")
                break
            if self.lr_scheduler:
                self.lr_scheduler.step()

    def test(self):
        test_mae, test_rmse = self.evaluate("test")
        print(f"[Test] MAE={test_mae:.4f} RMSE={test_rmse:.4f}")
        if 'test_mae' not in self.history:
            self.history['test_mae'] = test_mae
            self.history['test_rmse'] = test_rmse
        return test_mae, test_rmse
