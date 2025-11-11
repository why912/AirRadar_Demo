import os
import numpy as np
import argparse

# 根据 pos.npy 简单生成一个占位的 assignment.npy 与 mask.npy
# 真实生产应使用地理方位角与距离分层生成，这里仅用于演示与跑通流程。


def gen_placeholder(datapath: str, sectors: int = 12):
    pos_path = os.path.join(datapath, 'pos.npy')
    if not os.path.exists(pos_path):
        raise FileNotFoundError('pos.npy not found in ' + datapath)
    pos = np.load(pos_path)
    num_nodes = pos.shape[0]

    assignment = np.zeros((num_nodes, num_nodes, sectors), dtype=np.float32)
    # 简单按照相对经度差划分扇区（占位逻辑，仅演示）
    for i in range(num_nodes):
        for j in range(num_nodes):
            angle_bin = ((pos[j,0] - pos[i,0]) > 0).astype(int)  # 2分法占位
            assignment[i, j, angle_bin % sectors] = 1.0
    mask = np.zeros((num_nodes, sectors), dtype=bool)

    return assignment, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='./data')
    ap.add_argument('--name', type=str, default='AIR_TINY')
    ap.add_argument('--pattern', type=str, default='50-200')
    ap.add_argument('--sectors', type=int, default=12)
    args = ap.parse_args()

    base = os.path.join(args.root, 'local_partition', args.pattern)
    os.makedirs(base, exist_ok=True)

    assignment, mask = gen_placeholder(os.path.join(args.root, args.name), args.sectors)
    np.save(os.path.join(base, 'assignment.npy'), assignment)
    np.save(os.path.join(base, 'mask.npy'), mask)

    print('Placeholder dartboard saved to', base)


if __name__ == '__main__':
    main()
