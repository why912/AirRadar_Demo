import os
import numpy as np
import pickle
import argparse

REQUIRED_FILES = [
    "train.npz", "val.npz", "test.npz",
    "train_history.npy", "val_history.npy", "test_history.npy",
    "val_nodes.npy", "test_nodes.npy", "pos.npy"
]

CLASS_ID_SPECS = {
    11: (0, 10),  # wind_direction_id
    12: (0, 17),  # weather_id
    13: (0, 23),  # hour_id
    14: (0, 6),   # weekday_id
}


def check_exists(base):
    miss = []
    for f in REQUIRED_FILES:
        if not os.path.exists(os.path.join(base, f)):
            miss.append(f)
    return miss


def load_npz(path):
    d = np.load(path)
    assert 'x' in d and 'y' in d, f"{path} missing x or y"
    return d['x'], d['y']


def check_shapes(x, y, history, num_nodes, input_dim, output_dim, seq_len):
    ok = True
    if x.ndim != 3 or x.shape[1] != num_nodes or x.shape[2] != input_dim:
        print(f"[FAIL] x shape {x.shape}, expect [*, {num_nodes}, {input_dim}]")
        ok = False
    if y.ndim != 3 or y.shape[1] != num_nodes or y.shape[2] != output_dim:
        print(f"[FAIL] y shape {y.shape}, expect [*, {num_nodes}, {output_dim}]")
        ok = False
    if history.ndim != 4 or history.shape[1] != num_nodes or history.shape[2] != seq_len or history.shape[3] != input_dim:
        print(f"[FAIL] history shape {history.shape}, expect [*, {num_nodes}, {seq_len}, {input_dim}]")
        ok = False
    return ok


def check_class_ids(arr, name):
    # arr: [..., 27]
    ok = True
    for idx, (l, r) in CLASS_ID_SPECS.items():
        vals = arr[..., idx]
        if np.any(vals != np.floor(vals)):
            print(f"[WARN] {name} channel {idx} has non-integer values")
        vmin, vmax = vals.min(), vals.max()
        if vmin < l or vmax > r:
            print(f"[FAIL] {name} channel {idx} out of range [{l},{r}], got [{vmin},{vmax}]")
            ok = False
    return ok


def check_nodes_file(path, num_nodes):
    nodes = np.load(path)
    if nodes.ndim != 1:
        print(f"[FAIL] {path} is not 1D array")
        return False
    if nodes.min() < 0 or nodes.max() >= num_nodes:
        print(f"[FAIL] {path} index out of [0,{num_nodes-1}]")
        return False
    return True


def check_dartboard(base_lp, num_nodes):
    patterns = ["50-200", "50-200-500", "50", "25-100-250"]
    for p in patterns:
        ap = os.path.join(base_lp, p, 'assignment.npy')
        mp = os.path.join(base_lp, p, 'mask.npy')
        if not (os.path.exists(ap) and os.path.exists(mp)):
            print(f"[WARN] dartboard pattern {p} missing files")
            continue
        A = np.load(ap)
        M = np.load(mp)
        if A.ndim != 3 or A.shape[0] != num_nodes or A.shape[1] != num_nodes:
            print(f"[FAIL] assignment {p} shape {A.shape}")
            return False
        if M.ndim != 2 or M.shape[0] != num_nodes or M.shape[1] != A.shape[2]:
            print(f"[FAIL] mask {p} shape {M.shape} not match assignment sectors {A.shape[2]}")
            return False
    return True


def check_adj(path, num_nodes):
    if not os.path.exists(path):
        print(f"[WARN] graph {path} not found (skipped)")
        return True
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, tuple) and len(obj) >= 3:
        adj = obj[2]
    else:
        adj = obj
    if adj.shape != (num_nodes, num_nodes):
        print(f"[FAIL] adj shape {adj.shape}, expect ({num_nodes},{num_nodes})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True, help='path to data/<DATASET_NAME>')
    parser.add_argument('--num_nodes', type=int, default=1085)
    parser.add_argument('--input_dim', type=int, default=27)
    parser.add_argument('--output_dim', type=int, default=11)
    parser.add_argument('--seq_len', type=int, default=24)
    args = parser.parse_args()

    base = args.datapath
    miss = check_exists(base)
    if miss:
        print('[FAIL] missing files:', miss)
        return

    x_tr, y_tr = load_npz(os.path.join(base, 'train.npz'))
    x_v, y_v = load_npz(os.path.join(base, 'val.npz'))
    x_te, y_te = load_npz(os.path.join(base, 'test.npz'))

    h_tr = np.load(os.path.join(base, 'train_history.npy'))
    h_v = np.load(os.path.join(base, 'val_history.npy'))
    h_te = np.load(os.path.join(base, 'test_history.npy'))

    ok = True
    ok &= check_shapes(x_tr, y_tr, h_tr, args.num_nodes, args.input_dim, args.output_dim, args.seq_len)
    ok &= check_shapes(x_v, y_v, h_v, args.num_nodes, args.input_dim, args.output_dim, args.seq_len)
    ok &= check_shapes(x_te, y_te, h_te, args.num_nodes, args.input_dim, args.output_dim, args.seq_len)

    ok &= check_class_ids(x_tr, 'x_train')
    ok &= check_class_ids(h_tr, 'history_train')

    ok &= check_nodes_file(os.path.join(base, 'val_nodes.npy'), args.num_nodes)
    ok &= check_nodes_file(os.path.join(base, 'test_nodes.npy'), args.num_nodes)

    base_lp = os.path.join(os.path.dirname(base), 'local_partition')
    ok &= check_dartboard(base_lp, args.num_nodes)

    graph_path = os.path.join(os.path.dirname(base), 'sensor_graph', 'adj_mx_air_tiny.pkl')
    ok &= check_adj(graph_path, args.num_nodes)

    print('[RESULT]', 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
