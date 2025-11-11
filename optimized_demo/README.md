# Optimized Demo for AirRadar (Ethics-Oriented)

本 Demo 展示在不修改原始代码文件的前提下，对 AirRadar 项目进行的工程与伦理优化：

## 目录结构

```text
optimized_demo/
  model/airRadar_plus.py        # 动态节点支持 + 统一 forward 接口 + 去硬编码
  utils/trainer_plus.py         # 改进训练器，早停、设备自适应、伦理回调
  ethics/evaluation.py          # 公平性、能耗估算、频域解释占位
  configs/config.yaml           # 示例配置
  train.py                      # 标准训练脚本（需原始数据）
  run_demo.py                   # 快速演示 + 伦理摘要输出
  README.md                     # 当前说明
  ETHICS_CHECKLIST.md           # 伦理检查清单
```

## 主要改进

- 去除 FftNet / AFNO 节点数硬编码（动态初始化 pos_embed）
- 统一模型调用：`forward(inputs, history, mask_nodes=None, pred_attr="PM25", g=None)`
- TrainerPlus 简化并确保与新模型参数契合；加入公平性回调扩展点
- 伦理工具：能耗粗估、分组公平性（占位）、频域能量分布摘要

## 使用方法

1. 准备原始数据：确保 `./data/AIR_TINY/` 下存在原项目所需的 npy/npz 文件与 dartboard 分区文件。
2. 安装依赖（示例）：

   ```powershell
   pip install torch numpy scipy pyyaml tensorboardX
   ```

   注：本 Demo 不再依赖 timm，已内置 DropPath 实现。

3. 运行快速演示：

   ```powershell
   python optimized_demo/run_demo.py
   ```

4. 正常训练：

   ```powershell
   python optimized_demo/train.py
   ```

## 伦理检查清单（摘要）

详见 `ETHICS_CHECKLIST.md`。

- 公平性：分区/密度分层误差监控
- 透明性：记录配置与版本；频域与注意力解释接口占位
- 能源与环境：能耗估算函数（后续可接入真实 Profiler）
- 责任边界：输出仅用于辅助，不替代官方监测

## 后续工作

- 引入不确定性估计（MC Dropout / Ensembles）
- 真实公平性分组（按地理/经济指标）
- 绿色训练：混合精度 + 频域模式裁剪 + 蒸馏
- 解释接口对接可视化前端

## 免责声明

本 Demo 为教学与研究示例，伦理评估工具均为占位实现，不可直接用于生产环境决策。
