import os
import yaml
import torch
from optimized_demo.model.airRadar_plus import AirRadarPlus
from optimized_demo.utils.trainer_plus import TrainerPlus
from src.utils.helper import get_dataloader, check_device, get_num_nodes
from optimized_demo.ethics.evaluation import slice_fairness


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config('optimized_demo/configs/config.yaml')
    datapath = os.path.join('./data', 'AIR_TINY')
    device = check_device(None)

    num_nodes = get_num_nodes('AIR_TINY')
    # 简化：直接使用原项目参数集部分字段
    model = AirRadarPlus(
        dropout=0.3,
        hidden_channels=cfg['hidden_channels'],
        mlp_expansion=cfg['mlp_expansion'],
        num_heads=cfg['num_heads'],
        dartboard=cfg['dartboard'],
        name=cfg['model_name'],
        dataset='AIR_TINY',
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

    # 公平性回调占位：无真实 meta 时跳过
    fairness_callbacks = [lambda p, l, split: slice_fairness(p, l, split, meta=None)]

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
        fairness_callbacks=fairness_callbacks
    )

    trainer.train()
    trainer.test()


if __name__ == '__main__':
    main()
