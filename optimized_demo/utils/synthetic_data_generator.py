import os
import numpy as np
import argparse

# 生成一个最小玩具数据集，便于跑通 pipeline

def gen_dirs(root, name):
    base = os.path.join(root, name)
    os.makedirs(base, exist_ok=True)
    lp = os.path.join(root, 'local_partition')
    for p in ['50-200','50-200-500','50','25-100-250']:
        os.makedirs(os.path.join(lp, p), exist_ok=True)
    os.makedirs(os.path.join(root, 'sensor_graph'), exist_ok=True)
    return base, lp


def make_npz(path, ns, nn, idim, odim):
    x = np.random.randn(ns, nn, idim).astype(np.float32)
    y = np.random.randn(ns, nn, odim).astype(np.float32)
    # 分类字段按范围设置为整数
    x[...,11] = np.random.randint(0, 11, size=(ns, nn))
    x[...,12] = np.random.randint(0, 18, size=(ns, nn))
    x[...,13] = np.random.randint(0, 24, size=(ns, nn))
    x[...,14] = np.random.randint(0, 7, size=(ns, nn))
    # 标签缺失用 0
    y[y < 0] = 0
    np.savez(path, x=x, y=y)


def make_history(path, ns, nn, sl, idim):
    h = np.random.randn(ns, nn, sl, idim).astype(np.float32)
    h[...,11] = np.random.randint(0, 11, size=(ns, nn, sl))
    h[...,12] = np.random.randint(0, 18, size=(ns, nn, sl))
    h[...,13] = np.random.randint(0, 24, size=(ns, nn, sl))
    h[...,14] = np.random.randint(0, 7, size=(ns, nn, sl))
    np.save(path, h)


def make_nodes(path, nn):
    idx = np.arange(nn)
    np.random.shuffle(idx)
    np.save(path, idx)


def make_pos(path, nn):
    # 伪造经纬度，均匀分布
    lng = np.linspace(70, 135, nn)
    lat = np.linspace(15, 55, nn)
    pos = np.stack([lng, lat], axis=1).astype(np.float32)
    np.save(path, pos)


def make_dartboard(lp, nn, sectors):
    for p in ['50-200','50-200-500','50','25-100-250']:
        A = np.zeros((nn, nn, sectors), dtype=np.float32)
        # 简单随机分区：为每个 (i,j) 随机指派一个 sector
        rnd = np.random.randint(0, sectors, size=(nn, nn))
        for s in range(sectors):
            A[:,:,s] = (rnd == s).astype(np.float32)
        M = np.zeros((nn, sectors), dtype=bool)
        np.save(os.path.join(lp, p, 'assignment.npy'), A)
        np.save(os.path.join(lp, p, 'mask.npy'), M)


def make_graph(path, nn):
    # 简单环形邻接
    adj = np.zeros((nn, nn), dtype=np.float32)
    for i in range(nn):
        adj[i, (i+1)%nn] = 1.0
        adj[i, (i-1)%nn] = 1.0
    import pickle
    with open(path, 'wb') as f:
        pickle.dump((None, None, adj), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='./data')
    ap.add_argument('--name', type=str, default='AIR_TINY_TOY')
    ap.add_argument('--num_nodes', type=int, default=64)
    ap.add_argument('--train', type=int, default=64)
    ap.add_argument('--val', type=int, default=16)
    ap.add_argument('--test', type=int, default=16)
    ap.add_argument('--input_dim', type=int, default=27)
    ap.add_argument('--output_dim', type=int, default=11)
    ap.add_argument('--seq_len', type=int, default=24)
    ap.add_argument('--sectors', type=int, default=8)
    args = ap.parse_args()

    base, lp = gen_dirs(args.root, args.name)

    make_npz(os.path.join(base,'train.npz'), args.train, args.num_nodes, args.input_dim, args.output_dim)
    make_npz(os.path.join(base,'val.npz'), args.val, args.num_nodes, args.input_dim, args.output_dim)
    make_npz(os.path.join(base,'test.npz'), args.test, args.num_nodes, args.input_dim, args.output_dim)

    make_history(os.path.join(base,'train_history.npy'), args.train, args.num_nodes, args.seq_len, args.input_dim)
    make_history(os.path.join(base,'val_history.npy'), args.val, args.num_nodes, args.seq_len, args.input_dim)
    make_history(os.path.join(base,'test_history.npy'), args.test, args.num_nodes, args.seq_len, args.input_dim)

    make_nodes(os.path.join(base,'val_nodes.npy'), args.num_nodes)
    make_nodes(os.path.join(base,'test_nodes.npy'), args.num_nodes)

    make_pos(os.path.join(base,'pos.npy'), args.num_nodes)

    make_dartboard(lp, args.num_nodes, args.sectors)

    make_graph(os.path.join(args.root,'sensor_graph','adj_mx_air_tiny.pkl'), args.num_nodes)

    print('Synthetic dataset generated at', base)


if __name__ == '__main__':
    main()
