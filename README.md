# 工程导论课程 AirRadar Demo

本仓库提供基于 AirRadar 的最小可运行 Demo，面向真实北京小时级空气数据，打通“数据构建→一致性校验→训练评估→可视化→伦理摘要”的端到端闭环。模型融合方向扇区先验与频域全局表征，支持动态节点与缺测掩码；工程侧提供标准化数据规范、可视化产出与能耗等伦理占位指标，便于在教学科研与城市治理中快速验证与复现。

提示：仓库包含 Git LFS 跟踪的 npy/npz 大文件（约 700MB）。，在克隆后需安装并启用 Git LFS 后才能正确获取数据。

<mark>注：本项目使用了一部分AI工具，如ChatGPT等，特此声明。</mark>

## 团队成员

- 组长：王宇航
  
- 组员1：王思涵
  
- 组员2：陈家煜
  
- 组员3：杜为旭
  

## 快速开始

1. 创建并激活虚拟环境，安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. 安装并初始化 Git LFS（首次在本机执行）

```powershell
git lfs install
git lfs pull
```

3. 一键运行 Demo（默认使用数据集 `data/AIR_BJ`，预测 PM2.5）

```powershell
python optimized_demo/run_demo.py
```

运行结束后，模型指标与可视化图片将输出到 `data/AIR_BJ/viz/`，并在控制台打印简要伦理摘要（能耗估计与频域能量切片）。

## 主要特性

- 结构折中：方向扇区先验（DS-MSA）+ 频域 AFNO（FftNet）兼顾局部方向性与全局依赖
- 动态节点与掩码鲁棒：位置编码可变、训练/评估掩码一致
- 数据标准化：提供机器可读 schema 与验证器，自动检查文件与形状/范围
- 可视化：训练曲线、时间序列 Pred vs True（支持幅度匹配与平滑）、空间渐变热力图
- 伦理占位：能耗粗估与频域能量切片，便于绿色 AI 与过拟合风险观察

## 数据与目录

- 北京数据 CSV：`beijing_20250101-20251108/`（已入库，便于复现）
- 标准化数据集：`data/AIR_BJ/`（train/val/test 及对应 history、pos、val/test 节点划分）
- 可视化输出：`data/AIR_BJ/viz/`（训练曲线与示例图）

说明：仓库当前 `pos.npy` 与图/扇区为占位构造，用于闭环验证；替换为真实经纬度与距离构图/分区将显著影响指标与地图效果。

## 配置项（示例）

默认配置位于 `optimized_demo/configs/config.yaml`，常用字段：

- `max_epochs`、`batch_size`：训练轮数与批大小
- `pred_attr`：预测目标（如 `PM25`）
- `learning_rate`、`early_stop_patience`：优化与早停
- `dataset_root`：数据根目录（默认 `data/AIR_BJ`）

可通过修改该文件或在 `run_demo.py` 中传参来调整实验。

## 模型与代码结构

核心入口与模块：

- Demo 入口：`optimized_demo/run_demo.py`
- 模型实现：`optimized_demo/model/airRadar_plus.py`
- 训练器：`optimized_demo/utils/trainer_plus.py`
- 数据构建与校验：`optimized_demo/utils/build_beijing_dataset.py`、`optimized_demo/utils/dataset_validator.py`
- 可视化：`optimized_demo/vis/visualize.py`
- 伦理占位：`optimized_demo/ethics/evaluation.py`

原始论文版代码仍保留在 `src/` 与 `experiments/` 目录，便于对照。

## 伦理与可持续性

本 Demo 在流程内嵌能耗估计与频域概览，建议在正式训练中进一步接入 profiler 与碳排放估算；公平性分析可按区域/密度/风向进行切片评估；发布时需明确责任边界与高不确定性提示，避免“虚假确定性”。详见 `ethics_report/AI_Ethics_项目报告.md` 与 `optimized_demo/DEMO_DETAIL.md`。

## 常见问题（FAQ）

- LFS 拉取失败/速度慢：确认已执行 `git lfs install`，网络受限时可改用 Release 附件或对象存储分发数据。
- 指标与图像不理想：占位的经纬度与构图导致，替换为真实地理先验后会改善；也可调整 `configs/config.yaml`。
- 显存不足：降低 batch size，或缩短序列长度与隐藏维度。

## 致谢

本项目基于公开研究复现并做小范围的工程改造，去除了 timm 等重依赖，保留 DropPath、动态位置编码与数据校验器等组件，特此致谢原项目。
