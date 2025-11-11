# AirRadar 数据集设计规范 (Dataset Design)

## 1. 目标与范围

为全国尺度或区域尺度空气质量推断模型 (AirRadar / AirRadarPlus) 提供统一、可扩展、可验证的数据格式，支持：

- 多污染物指标预测 (PM2.5, PM10, NO2, CO, O3, SO2 等)；
- 类别与时间、气象特征嵌入；
- 掩码节点重建/缺测鲁棒评估；
- 局部扇区 (dartboard) 空间方向性建模；
- 图结构与位置坐标支持潜在 GNN 或地理增强。

适用场景：研究与教学、概念验证 (PoC)、小规模生产探索。非目标：个人隐私数据、精细化排放清单管理。

## 2. 顶层目录结构

```text
data/
  <DATASET_NAME>/           # 例如 AIR_TINY
    train.npz
    val.npz
    test.npz
    train_history.npy
    val_history.npy
    test_history.npy
    val_nodes.npy
    test_nodes.npy
    pos.npy
  local_partition/
    50-200/ (dartboard模式0)
      assignment.npy
      mask.npy
    50-200-500/ (模式1)
      assignment.npy
      mask.npy
    50/ (模式2)
      assignment.npy
      mask.npy
    25-100-250/ (模式3)
      assignment.npy
      mask.npy
  sensor_graph/
    adj_mx_<dataset>.pkl
```

## 3. 文件详细规范

### 3.1 核心样本文件 (`train.npz`, `val.npz`, `test.npz`)

- 格式：Numpy NPZ，键包含：`x`, `y`
- 形状：
  - `x`: [num_samples, num_nodes, input_dim]
  - `y`: [num_samples, num_nodes, output_dim]
- 默认参数：`input_dim=27`, `output_dim=11`
- 通道布局 (x)：
  - 0..10 : 11 个连续数值特征 (污染物 + 气象连续量等)
  - 11..14 : 4 个分类特征整数ID
    - 11: wind_direction_id ∈ [0,10]
    - 12: weather_id ∈ [0,17]
    - 13: hour_id ∈ [0,23] (项目原 embedding 命名为 "day" 的历史遗留)
    - 14: weekday_id ∈ [0,6] (项目原 embedding 命名为 "hour" 的历史遗留)
  - 15..26 : 12 个连续数值特征 (其它气象或派生变量)
- 标签布局 (y)：至少前 6 通道为：
  - 0: PM2.5
  - 1: PM10
  - 2: NO2
  - 3: CO
  - 4: O3
  - 5: SO2
  - 6..10: 可扩展其他目标 (如空气指数、湿度等)
- 缺失值约定：标签缺失或无效位置填 ≤0 (如 0 或 -1)，训练/评估时将过滤 label>0 的位置计算 MAE/RMSE。
- 数据类型：float32；分类字段可以存为浮点但值必须是整数。

### 3.2 历史序列 (`*_history.npy`)

- 形状： [num_samples, num_nodes, seq_len, input_dim]
- 默认：`seq_len=24` (过去 24 小时)
- 布局与范围：与 x 完全一致。
- 对齐：同索引的 `history[i]` 对应 `x[i]`, `y[i]` 的预测时间点。

### 3.3 节点掩码文件 (`val_nodes.npy`, `test_nodes.npy`)

- 一维数组：长度 = candidate_node_count
- 元素：节点整数索引 ∈ [0, num_nodes-1]
- 用法：评估阶段按照 `mask_rate` 截取前 `floor(len*mask_rate)` 个索引作为被 mask 节点，模拟缺测重建。

### 3.4 坐标文件 (`pos.npy`)

- 形状：[num_nodes, 2] → [longitude, latitude]
- 经纬度十进制度，float32。
- 用途：潜在地理特征增强、扇区划分或可视化。

### 3.5 Dartboard 分区 (`assignment.npy`, `mask.npy`)

- `assignment.npy`: [num_nodes, num_nodes, num_sectors]
  - 第一个 num_nodes：查询节点 (Q)
  - 第二个 num_nodes：候选目标节点 (K/V)
  - 第三个维度 num_sectors：按方向或距离聚合的扇区数
  - 数值：大多为 0 或权重系数 (可二值或归一化)，模型中用 `einsum` 投影至 sector 表示。
- `mask.npy`: [num_nodes, num_sectors]，bool
  - True 表示该查询节点的该扇区在注意力中屏蔽 (填 -inf)。
- 不同模式 (50-200 等) 对应不同分区策略，可按距离阈值或方向桶定义。

### 3.6 图结构 (`adj_mx_<dataset>.pkl`)

- 内容：通常包含 `(sensor_ids, sensor_id_to_ind, adj_mx)`
- `adj_mx`: [num_nodes, num_nodes] float32，非负权重，0 表示无边。
- 用途：兼容 GNN filter_type 支撑；AirRadarPlus 可忽略但推荐保留。

## 4. 参数与一致性校验

| 名称 | 说明 | 约束 |
|------|------|------|
| num_nodes | 节点总数 | 所有文件的节点维一致 (x,y,history,pos,assignment,mask,adj) |
| input_dim | 输入特征维度 | =27；分类切片固定为 11:15 |
| output_dim | 输出标签维度 | ≥6 且前6通道为污染物顺序固定 |
| seq_len | 历史窗口长度 | =24 (或与配置一致) |
| 分类ID范围 | 风向/天气/小时/星期 | 严格在各自枚举范围内，否则嵌入越界 |
| 缺失标签编码 | 标签无效位置 | ≤0 |
| assignment维度 | 第三维 num_sectors | 与 mask 第二维一致 |
| 节点索引有效性 | val/test_nodes | 均在 [0, num_nodes-1] 且可重复但不推荐 |

## 5. 生成流程参考 (Pseudo Pipeline)

1. 原始监测数据拉取：逐小时污染物 + 气象指标 + 元数据 (风向、天气、经纬度等)。
2. 对齐时间轴：补齐缺失小时，缺测值标记并在标签中置 0。连续特征可选插值再打缺失标记。
3. 分类 id 映射：风向离散化、天气类别标准化、时间字段 (hour, weekday) 提取。
4. 构造 x / y / history：
   - x[t] 即当前时刻特征；history[t] = (t-seq_len .. t-1) 的窗口；y[t] 为目标污染物 (可与 x 重叠)。
5. 标准化：记录各污染物通道 (前 output_dim) 的均值/方差用于训练阶段的 scaler；history/x 在加载时标准化，y 保持原始量纲。
6. 生成 val_nodes / test_nodes：按节点总列表随机打乱或按密度分层后拼接。
7. Dartboard assignment：根据经纬度计算相对方向与距离，分桶 (例：距离环 × 方位角 sectors)，构造稀疏 0/1 或加权矩阵；生成对应 mask (无数据或空扇区置 True)。
8. 图结构 adj_mx：根据距离阈值或相似度 (例如高斯核) 构建邻接矩阵；序列化为 pkl。

## 6. 质量控制清单

- 形状检查：所有数组实际形状与 schema 一致。
- 分类合法性：全量分类字段值域统计不越界；异常值计数 < 阈值。
- 缺失标签处理：标签中 ≤0 的比例统计；异常分布报警。
- assignment 稀疏度：非零比率合理 (过稀无法聚合，过密失去分区意义)。
- 交叉一致性：val/test_nodes 索引在 pos 和 adj 中均存在。
- 训练/验证划分时间穿越检测：确保 history 不包含未来信息。

## 7. 伦理与合规嵌入

| 方面 | 要求 | 数据层面措施 |
|------|------|--------------|
| 公平性 | 不地区性偏差 | 分区域误差切片、低密度节点额外权重备选 |
| 透明性 | 可追溯 | 保存生成脚本参数与日志、版本号标记 |
| 隐私 | 无个体轨迹 | 仅公共传感器数据，不含个人定位 |
| 能源 | 低碳 | 支持子集/抽样训练、频域裁剪、蒸馏计划 |
| 责任 | 正确使用 | 缺测与高不确定性区域标记风险标签 |

## 8. 版本与命名规范

- 数据集版本：`<name>_v<MAJOR>.<MINOR>.<PATCH>`，例：`AIR_TINY_v1.0.0`
- 变更日志：更新通道顺序、节点增减、清洗规则变更、分区策略变更。
- 与模型绑定：模型训练日志记录数据版本；发布时输出 `{model_version, dataset_version}` 元组。

## 9. 示例最小配置 (Toy)

```text
num_nodes = 32
input_dim = 27
output_dim = 11
seq_len = 24
train_samples = 64
val_samples = 16

num_sectors = 12  # toy dartboard
```

## 10. 常见错误与处理建议

| 错误 | 影响 | 修复 |
|------|------|------|
| 分类ID越界 | 嵌入层报错 | 重映射或裁剪至合法范围 |
| pos 缺失 | 无法生成 assignment | 回填估算坐标或剔除相关节点 |
| assignment 与 mask 维度不符 | 前向失败 | 重新生成 mask，与 assignment.shape[-1] 对齐 |
| y 被标准化 | 评估指标失真 | 保持 y 原值，只标准化 x/history |
| 节点数不一致 | 训练报 shape mismatch | 统一 num_nodes，重新导出全部文件 |
| 缺失值未标记 | MAE 偏低不真实 | 将缺失标签置 0 或 ≤0 |
| Dartboard 过密/过稀 | 注意力退化 | 调整距离阈值与角度分辨率 |

## 11. 与 Demo 的接口映射

- `optimized_demo/train.py` 中 `get_dataloader(datapath, batch_size, output_dim, mask_rate)` 假设：
  - 存在 train/val/test.npz 与对应历史 `*_history.npy`
  - 分类字段索引固定为 11:15
  - `val_nodes.npy`, `test_nodes.npy` 用于评估 mask 节点集合
- `AirRadarPlus` 动态从 `data/local_partition/<pattern>/assignment.npy` 加载分区矩阵。

## 12. 后续扩展建议

- 增加不确定性标签文件：`uncertainty_mask.npy`
- 加入区域/行政区划映射：`region_id.npy` → 支持公平性切片。
- 提供分层时间粒度 (小时/日均) 双版本：`train_hourly.npz`, `train_daily.npz`。
- 频域特征预计算缓存：加速全局模块。

---

本规范用于指导数据准备、验证与复现，若生产环境接入需进一步补充安全审计与监测策略。
