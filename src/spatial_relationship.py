#!/usr/bin/env python3
"""
Adipose-to-plasma-cell spatial relationship analysis (NS-cHL + obesity, Visium).

Replaces the abundance-based co-localisation bar chart with two depth-robust,
WITHIN-SAMPLE spatial statistics:

  1. Distance-decay profile - plasma cell module score as a function of distance
     to the nearest adipocyte-positive spot, with bootstrap CI and a permutation
     null band.
  2. Lee's L - bivariate spatial association between the adipocyte score and the
     plasma cell module, with a permutation p-value.

Both compare bins/neighbourhoods inside one sample, so differences in library
depth between samples do not drive the result.

Usage:
    python3 spatial_relationship.py --sample O2 \
        --h5 O2_filtered_feature_bc_matrix.h5 \
        --pos O2_tissue_positions_list.csv \
        --sf  O2_scalefactors_json.json
"""

import argparse, json
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree

# --- marker definitions -------------------------------------------------------
ADIPO = ["FABP4", "G0S2", "PLIN1"]
PLASMA = ["IGKC", "IGHG1", "MZB1", "XBP1", "JCHAIN"]

# --- gating rules (declared a priori, applied identically to every sample) -----
MIN_ADIPO_SPOTS = 50      # samples below this are excluded from the analysis
ADIPO_TOP_FRAC = 0.10     # adipocyte score must be in the top decile
ADIPO_MIN_GENES = 2       # ...AND >=1 count in at least 2 of the 3 markers
SPOT_UM = 55.0            # Visium spot diameter in micrometres
BIN_EDGES = np.arange(0, 901, 100)   # distance rings, micrometres
N_PERM = 1000
N_BOOT = 1000
KNN = 6                   # neighbours for Lee's L spatial weights
SEED = 0


def load_sample(h5_path, pos_path, sf_path):
    """Return (spot dataframe, log-normalised expression getter, um_per_px)."""
    with open(sf_path) as fh:
        sf = json.load(fh)
    um_per_px = SPOT_UM / sf["spot_diameter_fullres"]

    pos = pd.read_csv(
        pos_path, header=None,
        names=["barcode", "in_tissue", "array_row", "array_col",
               "pxl_row_fullres", "pxl_col_fullres"])

    with h5py.File(h5_path, "r") as fh:
        barcodes = fh["matrix/barcodes"][:].astype(str)
        genes = fh["matrix/features/name"][:].astype(str)
        mat = csc_matrix(
            (fh["matrix/data"][:], fh["matrix/indices"][:], fh["matrix/indptr"][:]),
            shape=fh["matrix/shape"][:])

    idx = {g: i for i, g in enumerate(genes)}
    bmap = {b: i for i, b in enumerate(barcodes)}

    spots = pos[pos["in_tissue"] == 1].copy()
    spots["mi"] = spots["barcode"].map(bmap)
    spots = spots.dropna(subset=["mi"])
    spots["mi"] = spots["mi"].astype(int)
    cols = spots["mi"].values

    # per-spot depth normalisation: CP10K + log1p (removes library-size effects)
    totals = np.asarray(mat[:, cols].sum(axis=0)).ravel()
    totals[totals == 0] = 1.0

    def raw(gene):
        if gene not in idx:
            return np.zeros(len(spots))
        return mat[idx[gene], :].toarray().ravel()[cols]

    def lognorm(gene):
        return np.log1p(raw(gene) / totals * 1e4)

    spots["x_um"] = spots["pxl_col_fullres"].values * um_per_px
    spots["y_um"] = spots["pxl_row_fullres"].values * um_per_px
    return spots, raw, lognorm, um_per_px


def module_score(lognorm, genes):
    """Mean of per-gene robustly scaled (99th pct) log-normalised expression."""
    cols = []
    for g in genes:
        v = lognorm(g)
        hi = np.percentile(v, 99)
        cols.append(v / hi if hi > 0 else v)
    return np.mean(cols, axis=0)


def adipocyte_mask(raw, adip_score):
    """Absolute rule: top-decile score AND corroborated by >=2 of 3 markers."""
    detected = np.sum([raw(g) > 0 for g in ADIPO], axis=0)
    cutoff = np.quantile(adip_score, 1 - ADIPO_TOP_FRAC)
    return (adip_score >= cutoff) & (detected >= ADIPO_MIN_GENES)


def distance_profile(xy, mask, values, rng):
    """Mean `values` per distance-to-adipose ring, with bootstrap CI and null."""
    tree = cKDTree(xy[mask])
    dist, _ = tree.query(xy, k=1)

    centres, obs, lo, hi = [], [], [], []
    for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        sel = (dist >= a) & (dist < b) & ~mask
        centres.append((a + b) / 2)
        if sel.sum() < 10:
            obs.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        v = values[sel]
        obs.append(v.mean())
        boot = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(N_BOOT)]
        lo.append(np.percentile(boot, 2.5)); hi.append(np.percentile(boot, 97.5))

    # permutation null: reassign the adipocyte labels at random, same count
    null = np.full((N_PERM, len(centres)), np.nan)
    n_ad = int(mask.sum())
    for p in range(N_PERM):
        perm = np.zeros(len(xy), dtype=bool)
        perm[rng.choice(len(xy), size=n_ad, replace=False)] = True
        d, _ = cKDTree(xy[perm]).query(xy, k=1)
        for j, (a, b) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
            sel = (d >= a) & (d < b) & ~perm
            if sel.sum() >= 10:
                null[p, j] = values[sel].mean()

    return dict(centre=np.array(centres), obs=np.array(obs),
                ci_lo=np.array(lo), ci_hi=np.array(hi),
                null_lo=np.nanpercentile(null, 2.5, axis=0),
                null_hi=np.nanpercentile(null, 97.5, axis=0),
                dist=dist)


def lees_l(xy, x, y, rng):
    """Lee's L bivariate spatial association + two-sided permutation p."""
    _, nn = cKDTree(xy).query(xy, k=KNN + 1)
    nn = nn[:, 1:]                       # drop self
    w = np.full(nn.shape, 1.0 / KNN)     # row-standardised weights

    def stat(a, b):
        az, bz = a - a.mean(), b - b.mean()
        la, lb = (az[nn] * w).sum(1), (bz[nn] * w).sum(1)
        denom = np.sqrt((az ** 2).sum() * (bz ** 2).sum())
        return len(a) * (la * lb).sum() / (w.sum(1) ** 2).sum() / denom

    obs = stat(x, y)
    null = np.array([stat(x, rng.permutation(y)) for _ in range(N_PERM)])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)
    return obs, p


def run(sample, h5, pos, sf):
    rng = np.random.default_rng(SEED)
    spots, raw, lognorm, um_px = load_sample(h5, pos, sf)
    xy = spots[["x_um", "y_um"]].values

    adip = module_score(lognorm, ADIPO)
    plas = module_score(lognorm, PLASMA)
    mask = adipocyte_mask(raw, adip)

    res = dict(sample=sample, n_spots=len(spots), n_adipo=int(mask.sum()),
               um_per_px=um_px,
               detection={g: float((raw(g) > 0).mean() * 100) for g in ADIPO + PLASMA})

    if mask.sum() < MIN_ADIPO_SPOTS:
        res["excluded"] = (f"only {mask.sum()} adipocyte-positive spots "
                           f"(pre-specified minimum {MIN_ADIPO_SPOTS})")
        return res, None

    res["lees_L"], res["lees_p"] = lees_l(xy, adip, plas, rng)
    profile = distance_profile(xy, mask, plas, rng)
    res["excluded"] = None
    return res, profile


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--sf", required=True)
    a = ap.parse_args()

    res, prof = run(a.sample, a.h5, a.pos, a.sf)
    print(f"\n{res['sample']}: {res['n_spots']:,} spots | "
          f"{res['n_adipo']} adipocyte-positive | {res['um_per_px']:.2f} um/px")
    print("  detection %:", {k: round(v, 1) for k, v in res["detection"].items()})
    if res["excluded"]:
        print(f"  EXCLUDED - {res['excluded']}")
    else:
        print(f"  Lee's L = {res['lees_L']:.3f} (p = {res['lees_p']:.4f})")
        print("  distance profile (um : plasma module):")
        for c, o in zip(prof["centre"], prof["obs"]):
            print(f"    {c:6.0f} : {o:.4f}" if np.isfinite(o) else f"    {c:6.0f} :    n/a")
