import os
import csv
import argparse
import numpy as np
from datetime import datetime, timedelta
import math
import pickle

NUMERIC_TYPES_1H = ["PM2.5","PM10","NO2","CO","O3","SO2","AQI"]
NUMERIC_TYPES_24H = ["PM2.5_24h","PM10_24h","NO2_24h","CO_24h","O3_24h","SO2_24h"]

# Mapping for y target channels
Y_ORDER = ["PM2.5","PM10","NO2","CO","O3","SO2"]


def parse_csv_file(path):
    """Parse a daily CSV file. Return list of (date_str, hour, type, values_dict)."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # header: date, hour, type, station1, station2, ...
        if len(header) < 4:
            return rows, []
        stations = header[3:]
        for r in reader:
            if not r or len(r) < 4:
                continue
            date_str = r[0].strip()
            hour = int(r[1])
            typ = r[2].strip()
            vals = r[3:]
            vdict = {}
            for idx, s in enumerate(stations):
                try:
                    v = float(vals[idx]) if vals[idx] != '' else float('nan')
                except Exception:
                    v = float('nan')
                vdict[s] = v
            rows.append((date_str, hour, typ, vdict))
    return rows, stations


def collect_timeseries(src_dir):
    """Collect all rows from multiple CSVs, grouped by timestamp and type."""
    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith('.csv')]
    files.sort()
    data = {}
    all_stations = None
    for fp in files:
        rows, stations = parse_csv_file(fp)
        if all_stations is None:
            all_stations = stations
        else:
            # intersect stations to be safe if mismatch
            all_stations = [s for s in all_stations if s in stations]
        for (ds, h, typ, vdict) in rows:
            key = (ds, h)
            if key not in data:
                data[key] = {}
            data[key][typ] = vdict
    return data, all_stations


def build_arrays(data, stations, seq_len=24, input_dim=27, output_dim=11):
    # sort timestamps
    def to_dt(ds, h):
        # date format yyyyMMdd, hour 1..24
        dt = datetime.strptime(ds, '%Y%m%d') + timedelta(hours=h-1)
        return dt
    keys = sorted(list(data.keys()), key=lambda k: to_dt(k[0], k[1]))
    T = len(keys)
    N = len(stations)
    # build per-timestamp feature and label for each node
    # x_t: [N, 27], y_t: [N, 11]
    xs = []
    ys = []
    hours = []
    weekdays = []
    for (ds, h) in keys:
        rec = data.get((ds, h), {})
        # numeric channels 0..10: fill with 1h and 24h features (pad if missing)
        x_num = np.zeros((N, 11), dtype=np.float32)
        # 1h types slots: 0..6 for PM2.5,PM10,NO2,CO,O3,SO2,AQI
        for i, typ in enumerate(NUMERIC_TYPES_1H):
            if typ in rec:
                for j, s in enumerate(stations):
                    v = rec[typ].get(s, float('nan'))
                    if not math.isfinite(v):
                        v = 0.0
                    x_num[j, i] = v
        # 24h types slots: 7..11 use first five from 24h list (will pad to 11 total)
        for i, typ in enumerate(NUMERIC_TYPES_24H):
            if i >= 4:  # we need at most 4 to fill up to 11
                break
            if typ in rec:
                for j, s in enumerate(stations):
                    v = rec[typ].get(s, float('nan'))
                    if not math.isfinite(v):
                        v = 0.0
                    x_num[j, 7 + i] = v
        # categorical 11..14: wind(0), weather(0), hour, weekday
        dt = datetime.strptime(ds, '%Y%m%d')
        hour_id = (h - 1) % 24
        weekday_id = dt.weekday()  # 0=Mon..6=Sun
        x_cat = np.zeros((N, 4), dtype=np.int64)
        x_cat[:, 0] = 0  # wind unknown
        x_cat[:, 1] = 0  # weather unknown
        x_cat[:, 2] = hour_id
        x_cat[:, 3] = weekday_id
        # tail 15..26 zeros
        x_tail = np.zeros((N, 12), dtype=np.float32)
        # assemble
        x = np.concatenate([x_num, x_cat.astype(np.float32), x_tail], axis=1)
        xs.append(x)
        # labels y: channels 0..5 in Y_ORDER
        y = np.zeros((N, output_dim), dtype=np.float32)
        for i, typ in enumerate(Y_ORDER):
            if typ in rec:
                for j, s in enumerate(stations):
                    v = rec[typ].get(s, float('nan'))
                    if not math.isfinite(v) or v <= 0:
                        v = 0.0
                    y[j, i] = v
        ys.append(y)
        hours.append(hour_id)
        weekdays.append(weekday_id)
    xs = np.stack(xs, axis=0)  # [T, N, 27]
    ys = np.stack(ys, axis=0)  # [T, N, output_dim]
    # build samples with history windows
    samples_x = []
    samples_y = []
    samples_hist = []
    for t in range(seq_len, T):
        hist = xs[t-seq_len:t]              # [24, N, 27]
        curx = xs[t]                        # [N, 27]
        cury = ys[t]                        # [N, output_dim]
        samples_x.append(curx)
        samples_y.append(cury)
        samples_hist.append(hist)
    X = np.stack(samples_x, axis=0)                    # [S, N, 27]
    Y = np.stack(samples_y, axis=0)                    # [S, N, output_dim]
    H = np.stack(samples_hist, axis=0)                 # [S, 24, N, 27]
    H = np.transpose(H, (0, 2, 1, 3))                  # [S, N, 24, 27]
    return X, Y, H, [to_dt(*k) for k in keys[seq_len:]]


def split_save(out_root, dataset_name, X, Y, H, stations, time_index, train_ratio=0.7, val_ratio=0.15):
    base = os.path.join(out_root, dataset_name)
    os.makedirs(base, exist_ok=True)
    S = X.shape[0]
    s_train = int(S * train_ratio)
    s_val = int(S * val_ratio)
    s_test = S - s_train - s_val
    # slices
    tr = slice(0, s_train)
    va = slice(s_train, s_train + s_val)
    te = slice(s_train + s_val, S)
    np.savez(os.path.join(base, 'train.npz'), x=X[tr], y=Y[tr])
    np.savez(os.path.join(base, 'val.npz'), x=X[va], y=Y[va])
    np.savez(os.path.join(base, 'test.npz'), x=X[te], y=Y[te])
    np.save(os.path.join(base, 'train_history.npy'), H[tr])
    np.save(os.path.join(base, 'val_history.npy'), H[va])
    np.save(os.path.join(base, 'test_history.npy'), H[te])
    # val/test nodes (shuffled)
    num_nodes = len(stations)
    idx = np.arange(num_nodes)
    np.random.shuffle(idx)
    np.save(os.path.join(base, 'val_nodes.npy'), idx)
    np.random.shuffle(idx)
    np.save(os.path.join(base, 'test_nodes.npy'), idx)
    # pos: synthetic grid around Beijing
    g = int(math.ceil(math.sqrt(num_nodes)))
    lng0, lat0 = 116.4, 39.9
    lngs = np.linspace(lng0 - 0.2, lng0 + 0.2, g)
    lats = np.linspace(lat0 - 0.2, lat0 + 0.2, g)
    pos = []
    for i in range(num_nodes):
        r, c = divmod(i, g)
        pos.append([lngs[c], lats[r]])
    pos = np.array(pos, dtype=np.float32)
    np.save(os.path.join(base, 'pos.npy'), pos)
    # graph adjacency (ring)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        adj[i, (i+1) % num_nodes] = 1.0
        adj[i, (i-1) % num_nodes] = 1.0
    graph_dir = os.path.join(out_root, 'sensor_graph')
    os.makedirs(graph_dir, exist_ok=True)
    with open(os.path.join(graph_dir, 'adj_mx_air_tiny.pkl'), 'wb') as f:
        pickle.dump((stations, {s:i for i,s in enumerate(stations)}, adj), f)
    # dartboard placeholders for all patterns
    lp_root = os.path.join(out_root, 'local_partition')
    for pat in ['50-200','50-200-500','50','25-100-250']:
        pdir = os.path.join(lp_root, pat)
        os.makedirs(pdir, exist_ok=True)
        # simple placeholder: random one-hot sector assignment
        sectors = 12
        A = np.zeros((num_nodes, num_nodes, sectors), dtype=np.float32)
        rnd = np.random.randint(0, sectors, size=(num_nodes, num_nodes))
        for s in range(sectors):
            A[:, :, s] = (rnd == s).astype(np.float32)
        M = np.zeros((num_nodes, sectors), dtype=bool)
        np.save(os.path.join(pdir, 'assignment.npy'), A)
        np.save(os.path.join(pdir, 'mask.npy'), M)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_dir', type=str, required=True, help='folder of beijing CSVs')
    ap.add_argument('--out_root', type=str, default='./data')
    ap.add_argument('--dataset_name', type=str, default='AIR_TINY')
    ap.add_argument('--seq_len', type=int, default=24)
    ap.add_argument('--input_dim', type=int, default=27)
    ap.add_argument('--output_dim', type=int, default=11)
    args = ap.parse_args()

    data, stations = collect_timeseries(args.src_dir)
    if not stations:
        raise RuntimeError('No stations found in CSV headers')
    X, Y, H, tidx = build_arrays(data, stations, seq_len=args.seq_len, input_dim=args.input_dim, output_dim=args.output_dim)
    base = split_save(args.out_root, args.dataset_name, X, Y, H, stations, tidx)
    print('Dataset built at', base, 'samples:', X.shape[0], 'nodes:', len(stations))


if __name__ == '__main__':
    main()
