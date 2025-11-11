"""快速演示脚本：
1. 载入配置与数据
2. 初始化改进模型与 TrainerPlus
3. 运行少量 epoch 并打印指标与占位伦理摘要
注意：需保证 ./data/AIR_TINY/* 数据已准备。
"""
import os
import torch
import sys
import os

# 确保项目根目录在 sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from optimized_demo.model.airRadar_plus import AirRadarPlus
from optimized_demo.utils.trainer_plus import TrainerPlus
from optimized_demo.ethics.evaluation import ethics_summary
from optimized_demo.vis.visualize import plot_training_curves, plot_pred_vs_true_series, plot_spatial_heatmap
from src.utils.helper import get_dataloader, check_device, get_num_nodes
import yaml


def load_cfg():
    with open('optimized_demo/configs/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_cfg()
    device = check_device(None)
    # 允许通过环境变量或配置切换数据集名称，默认使用新的北京数据集 AIR_BJ
    dataset_name = os.environ.get('AIR_DATASET', 'AIR_BJ')
    datapath = os.path.join('./data', dataset_name)
    # 动态节点数：读取 pos.npy 而不是使用硬编码映射
    try:
        from src.utils.helper import get_pos
        pos = get_pos(datapath)
        num_nodes = pos.shape[0]
    except Exception:
        # 回退：尝试旧逻辑（仅当使用内置 AIR_TINY 时）
        try:
            num_nodes = get_num_nodes(dataset_name)
        except Exception:
            raise RuntimeError(f"无法确定数据集 {dataset_name} 的节点数，请检查 pos.npy 是否存在")

    model = AirRadarPlus(
        dropout=0.3,
        hidden_channels=cfg['hidden_channels'],
        mlp_expansion=cfg['mlp_expansion'],
        num_heads=cfg['num_heads'],
        dartboard=cfg['dartboard'],
        name=cfg['model_name'],
    dataset=dataset_name,
        device=device,
        num_nodes=num_nodes,
        seq_len=24,
        horizon=24,
        input_dim=27,
        output_dim=11,
        mask_rate=cfg['mask_rate'],
        context_num=cfg['context_num'],
        block_num=cfg['block_num'],
        sparsity_threshold=cfg['sparsity_threshold']
    ).to(device)

    data, mask_nodes = get_dataloader(datapath, cfg['batch_size'], 11, cfg['mask_rate'])
    trainer = TrainerPlus(
        model=model,
        data=data,
        mask_nodes=mask_nodes,
        base_lr=cfg['base_lr'],
        steps=cfg['steps'],
        lr_decay_ratio=cfg['lr_decay_ratio'],
        max_epochs=cfg['max_epochs'],
        patience=cfg['patience'],
        pred_attr=cfg['pred_attr'],
    )

    trainer.train()
    mae, rmse = trainer.test()
    # 可视化输出目录
    vis_dir = os.path.join(datapath, 'viz')
    os.makedirs(vis_dir, exist_ok=True)
    # 1) 训练曲线
    if hasattr(trainer, 'history'):
        plot_training_curves(trainer.history, vis_dir)
    # 2) 采样一批 test，保存一段时间序列的 Pred vs True（取某个节点）
    # 为了简单，这里从 loader 中取一个 batch 进行预测
    test_loader = data['test_loader']
    batch = next(iter(test_loader))
    X, Y, H = [t.to(device) for t in batch]
    with torch.no_grad():
        pred, mask = model(X, H, mask_nodes['test'], cfg['pred_attr'])
    # 反标准化（与 trainer._inv_transform 一致，仅对通道0/PM25 做）
    pred_np = pred.cpu().numpy()
    Y_np = Y[..., 0:1].cpu().numpy()
    # 取第一个 batch 内的全部样本拼接时间序列（近似演示）
    # 将时间序列也用反标准化的尺度，以便与真实值幅度一致
    scaler0 = data['scalers'][0]
    series_pred_raw = pred_np[:, 0, 0]
    series_true_raw = Y_np[:, 0, 0]
    series_pred_inv = scaler0.inverse_transform(series_pred_raw)
    # True 本身已为原尺度（数据管线未对 y 做标准化），直接使用
    plot_pred_vs_true_series(
        series_pred_inv,
        series_true_raw,
        os.path.join(vis_dir, 'pred_vs_true_timeseries_node0.png'),
        title='Pred vs True (Node 0)',
        match_amplitude=True,
        smooth_win=9,
    )
    # 3) 空间散点图（取该 batch 第一条样本的所有节点）
    sample0_pred = pred_np[0, :, 0]
    # 还原到原始尺度
    scaler0 = data['scalers'][0]
    sample0_pred_inv = scaler0.inverse_transform(sample0_pred)
    try:
        from src.utils.helper import get_pos
        pos_std = get_pos(datapath)
        # 以三角剖分插值绘制渐变热力图风格
        plot_spatial_heatmap(pos_std, sample0_pred_inv, os.path.join(vis_dir, 'spatial_pred_sample0.png'), title='Spatial Prediction Heatmap (Sample 0)')
    except Exception:
        pass
    # 构建伦理摘要（占位）
    preds = torch.randn(100, device=device)
    labels = torch.randn(100, device=device)
    summary = ethics_summary(num_samples=len(data['train_loader'].dataset), epochs=cfg['max_epochs'], device=str(device), preds=preds, labels=labels)
    print('[EthicsSummary]', summary)
    print(f'[Viz] 图像已输出到: {vis_dir}')


if __name__ == '__main__':
    main()
