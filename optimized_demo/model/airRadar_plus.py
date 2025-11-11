import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from src.base.model import BaseModel
from src.layers.embedding import AirEmbedding
from src.models.afno.afno1d import AFNO1D

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def _trunc_normal_(tensor, std=0.02):
    with torch.no_grad():
        return tensor.normal_(mean=0.0, std=std)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SpatialAttention(nn.Module):
    def __init__(self, dim, heads, assignment, mask, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.assignment = assignment
        self.mask = mask
        self.num_sectors = assignment.shape[-1]
        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.relative_bias = nn.Parameter(torch.randn(heads, 1, self.num_sectors))
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        pre_kv = torch.einsum("bnc,mnr->bmrc", x, self.assignment).reshape(-1, self.num_sectors, C)
        pre_q = x.reshape(-1, 1, C)
        q = self.q_linear(pre_q).reshape(B * N, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = (
            self.kv_linear(pre_kv)
            .reshape(B * N, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.reshape(B, N, self.num_heads, 1, self.num_sectors) + self.relative_bias
        mask = self.mask.reshape(1, N, 1, 1, self.num_sectors)
        attn = (
            attn.masked_fill_(mask, float("-inf")).reshape(B * N, self.num_heads, 1, self.num_sectors).softmax(dim=-1)
        )
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.drop(x)
        return x


class DS_MSA(nn.Module):
    def __init__(self, dim, heads, mlp_dim, assignment, mask, dropout=0.0, depth=1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SpatialAttention(dim, heads=heads, assignment=assignment, mask=mask, dropout=dropout),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class AFNOBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, sparsity_threshold=0.01):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = AFNO1D(hidden_size=dim, num_blocks=1, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=1, hidden_size_factor=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), drop)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        return x + residual


class FftNet(nn.Module):
    def __init__(self, embed_dim, depth=2, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.0, sparsity_threshold=0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.drop = nn.Dropout(p=drop_rate)
        # pos_embed 动态按 num_nodes 初始化（在 reset_parameters 中完成）
        self.register_parameter("pos_embed", None)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [AFNOBlock(embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], sparsity_threshold=sparsity_threshold) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def reset_parameters(self, num_nodes: int):
        pe = nn.Parameter(torch.zeros(1, num_nodes, self.embed_dim))
        _trunc_normal_(pe, std=0.02)
        self.pos_embed = pe

    def forward(self, x):
        if self.pos_embed is None or self.pos_embed.shape[1] != x.shape[1]:
            # 若未初始化或节点数变化，则重新初始化位置编码
            self.reset_parameters(x.shape[1])
        x = x + self.pos_embed
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class Casual(nn.Module):
    def __init__(self, input_dim, expert_hidden_dim, expert_output_dim, num_experts, output_dim):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, expert_hidden_dim, expert_output_dim) for _ in range(num_experts)])
        self.gating = nn.Sequential(nn.Linear(input_dim, num_experts), nn.Softmax(dim=1))
        self.out = nn.Linear(expert_output_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        b, n, c = x.shape
        x2d = x.reshape(b * n, c)
        gate = self.gating(x2d)
        expert_outs = [expert(x2d) for expert in self.experts]
        weighted = torch.stack([gate[:, i].unsqueeze(1) * expert_outs[i] for i in range(len(self.experts))], dim=2).sum(dim=2)
        return self.out(weighted).reshape(b, n, self.output_dim)


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SpatialEmbeddingBlock(nn.Module):
    def __init__(self, hidden_channels, mlp_expansion, num_heads, assignment, mask, dropout, sparsity_threshold):
        super().__init__()
        dim = hidden_channels * 2
        self.local = DS_MSA(dim, heads=num_heads, mlp_dim=hidden_channels * mlp_expansion, assignment=assignment, mask=mask, dropout=dropout, depth=1)
        self.global_net = FftNet(embed_dim=dim, depth=1, sparsity_threshold=sparsity_threshold)
        self.out = nn.Linear(dim * 2, dim)

    def forward(self, x):
        local_x = self.local(x)
        global_x = self.global_net(x)
        return self.out(torch.cat([local_x, global_x], dim=-1))


class AirRadarPlus(BaseModel):
    def __init__(self, dropout=0.3, hidden_channels=32, mlp_expansion=2, num_heads=2, dartboard=0, mask_rate=0.5, context_num=2, block_num=2, sparsity_threshold=0.01, **args):
        super().__init__(**args)
        self.dropout = dropout
        self.embedding_air = AirEmbedding()
        self._mask_rate = mask_rate
        self.block_num = block_num
        # mask_token 维度应与原始输入维度一致（27）
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.input_dim))
        self.start_mlp = nn.Linear(self.input_dim, hidden_channels)
        self.start_conv = nn.Conv2d(self.input_dim, hidden_channels, kernel_size=(1, 1))
        # 将时间维(seq_len)聚合为1
        self.temporal_mlp = nn.Linear(self.seq_len, 1)
        # temporal 分支的通道压缩：数值(11)+嵌入(15)+尾部(12)=38 -> 压缩至 input_dim(27)
        self.temporal_proj = nn.Linear(38, self.input_dim)
        # spatial 分支同样由 38 压缩到 27
        self.spatial_proj = nn.Linear(38, self.input_dim)

        # dartboard assignment/mask 从磁盘读取，保持与原项目一致
        self.get_dartboard_info(dartboard)

        self.spatial_blocks = nn.ModuleList([
            SpatialEmbeddingBlock(hidden_channels, mlp_expansion, num_heads, self.assignment, self.mask, dropout, sparsity_threshold)
            for _ in range(self.block_num)
        ])
        # MoE 输出维度应与模型输出维度对齐（例如 11），后续根据 pred_attr 选取需要的通道
        self.casual = Casual(hidden_channels * 2, hidden_channels * 2, hidden_channels * 2, context_num, self.output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)

    def get_dartboard_info(self, dartboard):
        import numpy as np
        dartboard_map = {0: "50-200", 1: "50-200-500", 2: "50", 3: "25-100-250"}
        path_assignment = f"data/local_partition/{dartboard_map[dartboard]}/assignment.npy"
        path_mask = f"data/local_partition/{dartboard_map[dartboard]}/mask.npy"
        self.assignment = torch.from_numpy(np.load(path_assignment)).float()
        self.mask = torch.from_numpy(np.load(path_mask)).bool()

    def encoding_mask_noise(self, x, mask_ratio=0.75, mask_nodes: Optional[List[int]] = None):
        B, N, D = x.shape
        device = x.device
        if mask_nodes is not None:
            mask = torch.ones([B, N], device=device)
            idx = torch.tensor(mask_nodes, device=device)
            mask[:, idx] = 0
            x_keeped = x * mask.unsqueeze(-1)
            mask_tokens = self.mask_token.to(device).expand(B, 1, D)
            out_x = x_keeped + (1 - mask.unsqueeze(-1)) * mask_tokens
            return out_x, mask
        # random masking
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_keeped = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.zeros([B, N], device=device)
        mask[:, :len_keep] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)
        pad = ids_restore.shape[1] + 1 - x_keeped.shape[1]
        mask_tokens = self.mask_token.to(device).repeat(x_keeped.shape[0], pad, 1)
        out_x = torch.cat([x_keeped, mask_tokens], dim=1)
        out_x = torch.gather(out_x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, out_x.shape[2]))
        return out_x, mask

    def forward(self, inputs, history, mask_nodes: Optional[List[int]] = None, pred_attr: str = "PM25", g=None):
        # inputs: [B, N, C], history: [B, N, T, C]
        device = inputs.device
        # move assignment/mask to device
        self.assignment = self.assignment.to(device)
        self.mask = self.mask.to(device)

        x_masked, mask = self.encoding_mask_noise(inputs, self._mask_rate, mask_nodes)
        history = history * mask.unsqueeze(2).unsqueeze(3)

        # 处理分类特征索引：
        # 原始 embedding 定义: embed_wdir(11), embed_weather(18), embed_day(24), embed_hour(7)
        # 数据构建时我们填充顺序: wind(0..10), weather(0..17), hour(0..23), weekday(0..6)
        # 但原 AirEmbedding 把第三个当作 "hour(24)" 第四个当作 "weekday(7)" (变量名有混淆)。
        # 因此需要对顺序进行重排: 当前 x_masked[...,11:15] = [wind, weather, hour, weekday]
        # AirEmbedding.forward 期待顺序: [wind, weather, hour(映射到 embed_day/24), weekday(映射到 embed_hour/7)]
        cat_feats = x_masked[..., 11:15].clone()
        cat_feats_hist = history[..., 11:15].clone()
        # 将 hour 放到位置2 (保持), weekday 到位置3 (保持), 但要截断越界取值
        cat_feats[..., 2] = torch.clamp(cat_feats[..., 2], 0, 23)
        cat_feats[..., 3] = torch.clamp(cat_feats[..., 3], 0, 6)
        cat_feats_hist[..., 2] = torch.clamp(cat_feats_hist[..., 2], 0, 23)
        cat_feats_hist[..., 3] = torch.clamp(cat_feats_hist[..., 3], 0, 6)
        x_embed = self.embedding_air(cat_feats.long())
        temporal_embed = self.embedding_air(cat_feats_hist.long())
        # categorical embedding输出维度为 3+4+3+5=15，与原数据数值段拼接：数值前11 + 嵌入15 + 尾部12 = 38
        # start_conv 期望输入通道 self.input_dim=27，需要对 temporal 分支做线性降维到 27
        x_temporal_full = torch.cat((history[..., :11], temporal_embed, history[..., 15:]), -1)
        x_temporal = self.temporal_proj(x_temporal_full)
        x_spatial_full = torch.cat((x_masked[..., :11], x_embed, x_masked[..., 15:]), -1)
        x_spatial = self.spatial_proj(x_spatial_full)

        x_temporal = self.start_conv(x_temporal.permute(0, 3, 2, 1)).permute(0, 3, 1, 2)
        x_temporal = self.temporal_mlp(x_temporal).squeeze(-1)
        x_spatial = self.start_mlp(x_spatial)
        x = torch.cat([x_spatial, x_temporal], dim=-1)

        for blk in self.spatial_blocks:
            x = blk(x)
        x = self.casual(x)

        if pred_attr == "PM25":
            x = x[..., 0].unsqueeze(-1)
        return x, mask
