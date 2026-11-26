"""
Scene-level pipeline: fused scenes -> sliding windows -> PointNet++ -> kerb suppression -> metrics.
Run from the repository root:
  python pipeline.py --checkpoint path/to/best_model.pth --fused_dir path/to/fused_scenes --out_dir pipeline_out
Useful options: --threshold 0.4, --reuse (reuse saved probabilities), --no_kerb, --only <scene substrings>, --save_clouds.
"""
import os, sys, glob, time, argparse, importlib, csv, datetime
import numpy as np
from scipy.spatial import cKDTree

CHECKPOINT_PATH = 'path/to/best_model.pth'
FUSED_DIR  = 'path/to/fused_scenes'
MODEL_NAME = 'pointnet2_sem_seg_color'
NUM_CLASSES = 2

NUM_POINTS = 4096
WIN_X, WIN_Y = 0.365, 6.0
STRIDE_X, STRIDE_Y = WIN_X / 2, 3.0
MIN_PTS = 64
BATCH = 8


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w')
    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)
    def flush(self):
        sys.__stdout__.flush(); self.f.flush()


def load_model(checkpoint, device):
    import torch
    sys.path.append(os.path.join(os.getcwd(), 'models'))
    MODEL = importlib.import_module(MODEL_NAME)
    ckpt = torch.load(checkpoint, weights_only=False, map_location=device)
    cfg = ckpt.get('model_cfg', {})
    print(f'model configuration: {cfg if cfg else "original model (no configuration stored)"}')
    clf = MODEL.get_model(NUM_CLASSES, **cfg).to(device)
    clf.load_state_dict(ckpt['model_state_dict'])
    clf.eval()
    return clf


def predict_chunks(clf, patch, device):
    import torch
    M = len(patch)
    n_chunks = int(np.ceil(M / NUM_POINTS))
    order = np.random.permutation(M)
    pad = n_chunks * NUM_POINTS - M
    if pad:
        order = np.concatenate([order, np.random.choice(M, pad, replace=True)])
    chunks = order.reshape(n_chunks, NUM_POINTS)
    psum = np.zeros(M, np.float64); cnt = np.zeros(M, np.float64)
    with torch.no_grad():
        for b in range(0, n_chunks, BATCH):
            idx = chunks[b:b + BATCH]
            x = patch[idx].copy()
            x[:, :, :3] -= x[:, :, :3].mean(axis=1, keepdims=True)
            x = torch.from_numpy(x).float().transpose(2, 1).to(device)
            logp, _ = clf(x)
            p = logp.exp()[..., 1].cpu().numpy()
            np.add.at(psum, idx.ravel(), p.ravel()); np.add.at(cnt, idx.ravel(), 1)
    return psum / np.maximum(cnt, 1)


def predict_scene(clf, cloud, device, y_window=None):
    xyz = cloud[:, :3].astype(np.float32)
    feats = np.hstack([xyz, cloud[:, 4:7].astype(np.float32)])
    psum = np.zeros(len(xyz), np.float64); votes = np.zeros(len(xyz), np.float64)
    x0s = np.arange(xyz[:, 0].min(), max(xyz[:, 0].max() - WIN_X, xyz[:, 0].min()) + STRIDE_X, STRIDE_X)
    if y_window is not None:
        y0s, win_y = np.array([float(y_window[0])]), float(y_window[1]) - float(y_window[0])
    else:
        y0s = np.arange(xyz[:, 1].min(), max(xyz[:, 1].max() - WIN_Y, xyz[:, 1].min()) + STRIDE_Y, STRIDE_Y)
        win_y = WIN_Y
    for x0 in x0s:
        in_x = (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + WIN_X)
        if not in_x.any():
            continue
        for y0 in y0s:
            idx = np.where(in_x & (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + win_y))[0]
            if len(idx) < MIN_PTS:
                continue
            psum[idx] += predict_chunks(clf, feats[idx], device)
            votes[idx] += 1
    return (psum / np.maximum(votes, 1)).astype(np.float32), votes.astype(np.float32)


def road_height(xyz, fit_mask, iters=6, clip=0.04):
    if fit_mask.sum() < 10:
        return np.zeros(len(xyz), np.float32)
    A = np.c_[xyz[:, 0], xyz[:, 1], np.ones(len(xyz))]
    keep = fit_mask.copy(); coef = np.zeros(3)
    for _ in range(iters):
        if keep.sum() < 10:
            break
        coef, *_ = np.linalg.lstsq(A[keep], xyz[keep, 2], rcond=None)
        keep = fit_mask & (np.abs(xyz[:, 2] - A @ coef) < clip)
    return (xyz[:, 2] - A @ coef).astype(np.float32)


def suppress_kerbs(pred, xy, height, scored, kerb_h, kerb_r):
    out = pred.copy()
    wall = scored & (height > kerb_h)
    if wall.any() and pred.any():
        idx = np.where(pred)[0]
        d, _ = cKDTree(xy[wall]).query(xy[idx])
        out[idx[d <= kerb_r]] = False
        out[wall] = False
    return out


HYST_CELL, HYST_CLOSE = 0.05, 2


def hysteresis(low, seed, xy, min_seeds=20):
    from scipy import ndimage as ndi
    if not low.any():
        return seed.copy()
    x0, y0 = xy.min(0) - 1.0
    ix = ((xy[:, 0] - x0) / HYST_CELL).astype(int); iy = ((xy[:, 1] - y0) / HYST_CELL).astype(int)
    occ = np.zeros((ix.max() + 2, iy.max() + 2), bool); occ[ix[low], iy[low]] = True
    p = HYST_CLOSE + 1
    closed = ndi.binary_closing(np.pad(occ, p), structure=np.ones((3, 3), bool), iterations=HYST_CLOSE)[p:-p, p:-p]
    grid, K = ndi.label(closed, structure=np.ones((3, 3)))
    comp = grid[ix, iy]
    ok = np.bincount(comp[seed], minlength=K + 1) >= min_seeds
    ok[0] = False
    return (low & ok[comp]) | seed


def from_counts(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1); biou = tn / max(tn + fp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, acc=(tp + tn) / max(n, 1), prec=prec, rec=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12), iou=iou, bg_iou=biou, miou=(iou + biou) / 2)


def score(pred, tgt):
    return from_counts(int(np.sum(pred & tgt)), int(np.sum(pred & ~tgt)),
                       int(np.sum(~pred & tgt)), int(np.sum(~pred & ~tgt)))


def sum_counts(ms):
    return from_counts(*(sum(m[k] for m in ms) for k in ('tp', 'fp', 'fn', 'tn')))


def process_scene(raw, prob, votes, a):
    xyz = raw[:, :3]
    coloured = raw[:, 4:7].sum(1) > 0
    scored = (votes > 0) & (coloured if not a.score_all else True)
    tgt = raw[:, 3] > 0
    height = road_height(xyz, coloured & (votes > 0))
    pred0 = prob >= a.threshold
    seed, low = pred0 & scored, None
    if a.hyst_low is not None:
        low = ((prob >= a.hyst_low) & scored) | seed
    if not a.no_kerb:
        seed = suppress_kerbs(seed, xyz[:, :2], height, scored, a.kerb_h, a.kerb_r)
        if low is not None:
            low = suppress_kerbs(low, xyz[:, :2], height, scored, a.kerb_h, a.kerb_r) | seed
    pred1 = hysteresis(low, seed, xyz[:, :2], a.hyst_min_seeds) if low is not None else seed
    m0, m1 = score(pred0[scored], tgt[scored]), score(pred1[scored], tgt[scored])
    return m0, m1, dict(pred=pred1, height=height, scored=scored, tgt=tgt)


def fmt(m):
    return (f"Acc {m['acc']*100:5.1f}  Prec {m['prec']*100:5.1f}  Rec {m['rec']*100:5.1f}  "
            f"F1 {m['f1']*100:5.1f}  IoU {m['iou']*100:5.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fused_dir', default=FUSED_DIR)
    ap.add_argument('--checkpoint', default=CHECKPOINT_PATH)
    ap.add_argument('--out_dir', default='./pipeline_out')
    ap.add_argument('--threshold', type=float, default=0.5, help='defect probability threshold')
    ap.add_argument('--kerb_h', type=float, default=0.08, help='metres above road plane that counts as kerb wall')
    ap.add_argument('--kerb_r', type=float, default=0.4, help='suppression radius around kerb wall (m)')
    ap.add_argument('--no_kerb', action='store_true', help='disable kerb suppression')
    ap.add_argument('--only', default='', help='comma-separated scene-name substrings')
    ap.add_argument('--reuse', action='store_true', help='reuse saved probabilities instead of running the model')
    ap.add_argument('--prob_dir', default=None, help='where probabilities are read/written (default OUT/probs)')
    ap.add_argument('--save_clouds', action='store_true', help='save per-point results for the classification step')
    ap.add_argument('--score_all', action='store_true', help='also score points without camera colour')
    ap.add_argument('--hyst_low', type=float, default=None,
                    help='hysteresis: also accept points with probability >= this IF connected to a core >= --threshold')
    ap.add_argument('--hyst_min_seeds', type=int, default=20, help='core points a region needs before it may grow')
    ap.add_argument('--y_window', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                    help='run the model only inside this y-range (training strips use -7 -1); needs a model run, not --reuse')
    ap.add_argument('--seed', type=int, default=0, help='random seed (random centroids and chunking make runs differ slightly)')
    a = ap.parse_args()
    np.random.seed(a.seed)
    try:
        import torch as _torch
        _torch.manual_seed(a.seed)
    except ImportError:
        pass

    os.makedirs(a.out_dir, exist_ok=True)
    prob_dir = a.prob_dir or os.path.join(a.out_dir, 'probs'); os.makedirs(prob_dir, exist_ok=True)
    cloud_dir = os.path.join(a.out_dir, 'clouds')
    if a.save_clouds:
        os.makedirs(cloud_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(a.out_dir, 'run_log.txt'))

    print(f'Road-defect pipeline | {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'command  : {" ".join(sys.argv)}')
    print(f'threshold: {a.threshold} | kerb suppression: '
          f'{"OFF" if a.no_kerb else f"ON (H={a.kerb_h} m, R={a.kerb_r} m)"} | scoring: '
          f'{"all voted points" if a.score_all else "points with camera colour"}')
    if a.y_window:
        print(f'windows  : {WIN_X} m across x y in [{a.y_window[0]}, {a.y_window[1]}] (training-strip crop; only this region is predicted and scored), '
              f'{NUM_POINTS} pts/chunk, centred per chunk')
    else:
        print(f'windows  : {WIN_X} m across x {WIN_Y} m along Y, {NUM_POINTS} pts/chunk, centred per chunk')
    if a.hyst_low is not None:
        print(f'hysteresis: grow from cores >= {a.threshold} to points >= {a.hyst_low} (regions need >= {a.hyst_min_seeds} core points)')

    files = sorted(glob.glob(os.path.join(a.fused_dir, '*.npy')))
    if a.only:
        keys = a.only.split(',')
        files = [f for f in files if any(k in os.path.basename(f) for k in keys)]
    print(f'scenes   : {len(files)}\n')

    clf, device = None, None
    if not a.reuse:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'loading model on {device}: {a.checkpoint}')
        clf = load_model(a.checkpoint, device)

    rows, before, after = [], [], []
    t_start = time.time()
    for k, path in enumerate(files, 1):
        name = os.path.basename(path).replace('.npy', '')
        t0 = time.time()
        raw = np.load(path)
        pf = os.path.join(prob_dir, name + '_prob.npz')
        if a.reuse:
            if not os.path.exists(pf):
                print(f'[{k}/{len(files)}] {name}: no saved probabilities in {prob_dir}, skipped')
                continue
            d = np.load(pf); prob, votes = d['prob'], d['votes']
        else:
            xy = raw[raw[:, 4:7].sum(1) > 0][:, :2]
            if len(xy) > 100:
                w, V = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))
                if abs(V[0, 1]) > abs(V[1, 1]):
                    print(f'  WARNING {name}: principal axis is X, windows assume the road runs along Y')
            prob, votes = predict_scene(clf, raw, device, a.y_window)
            np.savez_compressed(pf, prob=prob, votes=votes)

        m0, m1, arr = process_scene(raw, prob, votes, a)
        if arr['scored'].sum() == 0:
            print(f'[{k}/{len(files)}] {name}: nothing scored, skipped'); continue
        gt = m1['tp'] + m1['fn']
        before.append(m0); after.append(m1)
        rows.append(dict(scene=name, session=name[:14], gt_defect_pts=gt, **{f'raw_{x}': m0[x] for x in
                    ('acc', 'prec', 'rec', 'f1', 'iou')}, **{x: m1[x] for x in ('acc', 'prec', 'rec', 'f1', 'iou', 'tp', 'fp', 'fn', 'tn')}))
        print(f'[{k}/{len(files)}] {name} | IoU {m0["iou"]*100:5.1f} -> {m1["iou"]*100:5.1f} | '
              f'Acc {m1["acc"]*100:5.1f}  Prec {m1["prec"]*100:5.1f}  Rec {m1["rec"]*100:5.1f} | {time.time()-t0:.0f}s')

        if a.save_clouds:
            out = np.column_stack([raw[:, :3], raw[:, 3], raw[:, 4:7], prob, arr['pred'].astype(np.float32),
                                   arr['height'], arr['scored'].astype(np.float32)]).astype(np.float32)
            np.save(os.path.join(cloud_dir, name + '_pred.npy'), out)

    if not rows:
        print('\nNo scenes were scored.'); return

    with open(os.path.join(a.out_dir, 'scene_metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    has_gt = [r for r in rows if r['gt_defect_pts'] > 0]
    G0, G1 = sum_counts(before), sum_counts(after)
    mean = lambda key: np.mean([r[key] for r in has_gt]) * 100

    print('\n' + '=' * 78)
    print(f'SUMMARY  ({len(rows)} scenes, defect class, threshold {a.threshold})')
    print('=' * 78)
    print('GLOBAL (all scored points pooled)')
    if a.hyst_low is None:
        print(f'  before kerb suppression : {fmt(G0)}')
        print(f'  after  kerb suppression : {fmt(G1)}' + ('   [suppression off]' if a.no_kerb else ''))
    else:
        print(f'  threshold only                   : {fmt(G0)}')
        print(f'  + kerb rule + hysteresis ({a.hyst_low}) : {fmt(G1)}' + ('   [kerb rule off]' if a.no_kerb else ''))
    print(f'  background IoU {G1["bg_iou"]*100:.1f} | mean IoU (2 classes) {G1["miou"]*100:.1f}')
    print(f'  points: TP {G1["tp"]:,}  FP {G1["fp"]:,}  FN {G1["fn"]:,}  TN {G1["tn"]:,}')
    print(f'PER-SCENE MEAN ({len(has_gt)} scenes with labelled defects)')
    print(f'  Acc {mean("acc"):.1f}  Prec {mean("prec"):.1f}  Rec {mean("rec"):.1f}  '
          f'F1 {mean("f1"):.1f}  IoU {mean("iou"):.1f}   (median IoU {np.median([r["iou"] for r in has_gt])*100:.1f})')

    print('\nBY CAPTURE SESSION (mean per-scene, after suppression)')
    print('  session         scenes |  Prec   Rec   IoU')
    for s in sorted({r['session'] for r in has_gt}):
        rs = [r for r in has_gt if r['session'] == s]
        print(f'  {s}  {len(rs):5d}  | {np.mean([r["prec"] for r in rs])*100:5.1f} '
              f'{np.mean([r["rec"] for r in rs])*100:5.1f} {np.mean([r["iou"] for r in rs])*100:5.1f}')

    worst = sorted(has_gt, key=lambda r: r['iou'])[:5]
    print('\nWEAKEST 5 SCENES')
    for r in worst:
        kind = 'FP-heavy' if r['fp'] > r['fn'] else 'FN-heavy'
        print(f'  {r["scene"]}  IoU {r["iou"]*100:5.1f}  Prec {r["prec"]*100:5.1f}  Rec {r["rec"]*100:5.1f}  ({kind})')

    print(f'\nfiles: {a.out_dir}/run_log.txt, scene_metrics.csv' + (f', clouds/ ({len(rows)} per-point files)' if a.save_clouds else ''))
    print(f'total time {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
