#!/usr/bin/env python3
"""
Reproduces every statistic reported in the manuscript.

Usage:
    python src/reproduce_statistics.py --data data/

Expects one directory per sample under --data, each containing the Space Ranger
outputs: filtered_feature_bc_matrix.h5, tissue_positions_list.csv and
scalefactors_json.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, h5py
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree

SAMPLES = ['C1', 'C2', 'O1', 'O2']
ANALYSED   = ['C1', 'C2', 'O2']       # O1 excluded for RNA quality
TARGET_UMI = 410                       # common depth for downsampling
SPOT_UM    = 55.0
SEED       = 0

ADIPO   = ['FABP4', 'G0S2', 'PLIN1']
PLASMA  = ['IGKC', 'IGHG1', 'MZB1', 'XBP1', 'JCHAIN']
CONTROL = ['ACTB', 'GAPDH', 'RPL13A', 'TMSB4X', 'EEF1A1']

MIN_ADIPO_SPOTS = 50
ADIPO_TOP_FRAC  = 0.10
ADIPO_MIN_GENES = 2
BIN_EDGES = np.arange(0, 901, 100)
N_PERM = N_BOOT = 1000
KNN = 6


def load(sample_dir):
    sf = json.load(open(sample_dir / 'scalefactors_json.json'))
    um_per_px = SPOT_UM / sf['spot_diameter_fullres']
    pos = pd.read_csv(sample_dir / 'tissue_positions_list.csv', header=None,
                      names=['barcode', 'in_tissue', 'array_row', 'array_col',
                             'pxl_row_fullres', 'pxl_col_fullres'])
    with h5py.File(sample_dir / 'filtered_feature_bc_matrix.h5', 'r') as fh:
        barcodes = fh['matrix/barcodes'][:].astype(str)
        genes = fh['matrix/features/name'][:].astype(str)
        mat = csc_matrix((fh['matrix/data'][:], fh['matrix/indices'][:],
                          fh['matrix/indptr'][:]), shape=fh['matrix/shape'][:])
    spots = pos[pos.in_tissue == 1].copy()
    spots['mi'] = spots.barcode.map({b: i for i, b in enumerate(barcodes)})
    spots = spots.dropna(subset=['mi']); spots['mi'] = spots.mi.astype(int)
    return dict(idx={g: i for i, g in enumerate(genes)},
                mat=mat[:, spots.mi.values].tocsc(),
                xy=np.c_[spots.pxl_col_fullres.values * um_per_px,
                         spots.pxl_row_fullres.values * um_per_px],
                n=len(spots))


def downsample(mat, rng, target=TARGET_UMI):
    """Binomial thinning to a common median depth."""
    umi = np.asarray(mat.sum(0)).ravel()
    p = np.minimum(1.0, target / np.maximum(umi, 1))
    co = mat.tocoo()
    return csc_matrix((rng.binomial(co.data.astype(int), p[co.col]),
                       (co.row, co.col)), shape=mat.shape)


def counts(d, gene, m=None):
    m = d['mat'] if m is None else m
    return (np.zeros(m.shape[1]) if gene not in d['idx']
            else np.asarray(m[d['idx'][gene]].todense()).ravel())


def lognorm(d, gene, m=None):
    m = d['mat'] if m is None else m
    tot = np.asarray(m.sum(0)).ravel(); tot[tot == 0] = 1
    return np.log1p(counts(d, gene, m) / tot * 1e4)


def module(d, genes, m=None):
    out = []
    for g in genes:
        v = lognorm(d, g, m); hi = np.percentile(v, 99)
        out.append(v / hi if hi > 0 else v)
    return np.mean(out, axis=0)


def moran(xy, v):
    _, nn = cKDTree(xy).query(xy, k=KNN + 1); nn = nn[:, 1:]
    z = v - v.mean()
    return np.nan if z.std() == 0 else (z * z[nn].mean(1)).sum() / (z ** 2).sum()


def lees_l(xy, x, y, rng):
    _, nn = cKDTree(xy).query(xy, k=KNN + 1); nn = nn[:, 1:]
    w = np.full(nn.shape, 1.0 / KNN)

    def stat(a, b):
        az, bz = a - a.mean(), b - b.mean()
        la, lb = (az[nn] * w).sum(1), (bz[nn] * w).sum(1)
        return len(a) * (la * lb).sum() / (w.sum(1) ** 2).sum() / \
            np.sqrt((az ** 2).sum() * (bz ** 2).sum())

    obs = stat(x, y)
    null = np.array([stat(x, rng.permutation(y)) for _ in range(N_PERM)])
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)


def residualise(v, depth):
    X = np.c_[np.ones_like(depth), depth]
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def adipocyte_mask(d, adip):
    det = np.sum([counts(d, g) > 0 for g in ADIPO], axis=0)
    return (adip >= np.quantile(adip, 1 - ADIPO_TOP_FRAC)) & (det >= ADIPO_MIN_GENES)


def distance_profile(xy, mask, values, rng):
    dist, _ = cKDTree(xy[mask]).query(xy, k=1)
    centres, obs, lo, hi = [], [], [], []
    for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        sel = (dist >= a) & (dist < b) & ~mask
        centres.append((a + b) / 2)
        if sel.sum() < 10:
            obs += [np.nan]; lo += [np.nan]; hi += [np.nan]; continue
        v = values[sel]; obs.append(v.mean())
        boot = [rng.choice(v, len(v), replace=True).mean() for _ in range(N_BOOT)]
        lo.append(np.percentile(boot, 2.5)); hi.append(np.percentile(boot, 97.5))
    null = np.full((N_PERM, len(centres)), np.nan)
    for p in range(N_PERM):
        perm = np.zeros(len(xy), bool)
        perm[rng.choice(len(xy), int(mask.sum()), replace=False)] = True
        dd, _ = cKDTree(xy[perm]).query(xy, k=1)
        for j, (a, b) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
            sel = (dd >= a) & (dd < b) & ~perm
            if sel.sum() >= 10:
                null[p, j] = values[sel].mean()
    return (np.array(centres), np.array(obs), np.array(lo), np.array(hi),
            np.nanpercentile(null, 2.5, 0), np.nanpercentile(null, 97.5, 0))


def main(data_dir):
    D = {}
    for code in SAMPLES:
        path = data_dir / code
        if not path.is_dir():
            sys.exit(f'No directory {path}')
        D[code] = load(path)
        D[code]['ds'] = downsample(D[code]['mat'], np.random.default_rng(SEED))

    print('\n=== Sequencing depth (Supplementary Table 1) ===')
    for c in SAMPLES:
        u = np.asarray(D[c]['mat'].sum(0)).ravel()
        g = np.asarray((D[c]['mat'] > 0).sum(0)).ravel()
        print(f'  {c}: {D[c]["n"]:>5} spots | median {np.median(u):>6.0f} UMI '
              f'| median {np.median(g):>5.0f} panel genes')

    print(f'\n=== Detection at matched depth ({TARGET_UMI} UMI/spot) ===')
    print(f'  {"gene":<9}' + ''.join(f'{c:>9}' for c in ANALYSED) + '   fold (O2 / max control)')
    for g in ['CCL22', 'IGKC', 'IGHG1', 'HLA-C', 'TXNIP', 'SFRP4',
              'COL1A2', 'COL3A1', 'PCOLCE', 'FOS', 'ACTB']:
        det = {c: 100 * (counts(D[c], g, D[c]['ds']) > 0).mean() for c in ANALYSED}
        fold = det['O2'] / max(det['C1'], det['C2'], 0.15)
        print(f'  {g:<9}' + ''.join(f'{det[c]:>8.1f}%' for c in ANALYSED) + f'   {fold:>6.1f}x')

    print("\n=== Moran's I at matched depth ===")
    for g in ['IGKC', 'IGHG1', 'IGLC1', 'SFRP4', 'CCL22', 'COL1A2']:
        v = {c: moran(D[c]['xy'], lognorm(D[c], g, D[c]['ds'])) for c in ANALYSED}
        print(f'  {g:<8}' + ''.join(f'{c}={v[c]:+.3f}  ' for c in ANALYSED))

    print('\n=== Adipose interface (Figure 2C-F) ===')
    for c in ANALYSED:
        d = D[c]
        depth = np.log10(np.asarray(d['mat'].sum(0)).ravel() + 1)
        adip, plas, ctrl = module(d, ADIPO), module(d, PLASMA), module(d, CONTROL)
        mask = adipocyte_mask(d, adip)
        print(f'  {c}: {mask.sum()} adipocyte-positive spots of {d["n"]}', end='')
        if mask.sum() < MIN_ADIPO_SPOTS:
            print('  -> excluded (pre-specified minimum 50)'); continue
        L, p = lees_l(d['xy'], residualise(adip, depth), residualise(plas, depth),
                      np.random.default_rng(SEED))
        Lc, _ = lees_l(d['xy'], adip, ctrl, np.random.default_rng(SEED))
        print(f"  |  Lee's L = {L:+.3f} (p = {p:.4f})  |  housekeeping control = {Lc:+.3f}")
        cen, obs, lo, hi, nlo, nhi = distance_profile(
            d['xy'], mask, residualise(plas, depth), np.random.default_rng(SEED))
        print('      distance (um) : plasma module [95% CI]   (* outside permutation null)')
        for x, o, a, b, na, nb in zip(cen, obs, lo, hi, nlo, nhi):
            if np.isfinite(o):
                print(f'      {x:>6.0f} : {o:+.4f} [{a:+.4f}, {b:+.4f}]'
                      f'{" *" if (o < na or o > nb) else ""}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('data'))
    main(ap.parse_args().data)
