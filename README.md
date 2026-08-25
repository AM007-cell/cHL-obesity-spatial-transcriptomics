# Spatial transcriptomics of classical Hodgkin lymphoma lymph nodes with and without obesity

Analysis code accompanying the manuscript.

## Contents

```
src/reproduce_statistics.py   reproduces every statistic reported in the manuscript
src/spatial_relationship.py   core module: depth matching, module scores, Moran's I,
                              Lee's L, distance-to-interface profiling
figures/                      scripts that generate each published figure panel
requirements.txt
```

## Data

Not included here. The Space Ranger outputs are not included here. Raw data are restricted and available from the corresponding author upon reasonable request, subject to institutional data sharing agreement and ethics approval. Place one directory per section under data/, each containing:. Place one
directory per sample under `data/`, each containing:

```
data/<CODE>/filtered_feature_bc_matrix.h5
data/<CODE>/tissue_positions_list.csv
data/<CODE>/scalefactors_json.json
data/<CODE>/tissue_lowres_image.png     # figures only
```

Directory names must be `C1`, `C2`, `O1`, `O2`. C1 and C2 are the non-obese
sections; O1 and O2 the obese sections. O1 is excluded from quantitative
comparisons for RNA quality but is read for the quality control summary.

## Running

```bash
pip install -r requirements.txt
python src/reproduce_statistics.py --data data/
```

Prints sequencing depth per section, detection frequencies at matched depth,
Moran's I, Lee's L with its housekeeping negative control, and the
distance-decay profile relative to the adipose interface.

## Analysis notes

Sequencing depth differs approximately fivefold between sections. All
between-sample comparisons are made after binomial downsampling to a common
median depth of 410 UMI per spot (`TARGET_UMI`), and a transcript is treated as
obesity-associated only when enriched at least twofold against **both**
non-obese sections at matched depth.

Module scores are linearly adjusted for per-spot log10 total UMI before spatial
analysis. The specificity of that adjustment is verified with a housekeeping
module used as a negative control; a strongly non-zero control value indicates a
residual RNA-quality gradient across the section rather than biological
association.

All random procedures use a fixed seed (`SEED = 0`). Permutation and bootstrap
procedures use 1,000 iterations.

Clustering, cluster annotation and cluster-level differential expression were
performed separately in Seurat and are not reproduced by this code; the
differential expression output is provided as Supplementary Table 4.

## Licence

MIT (see LICENSE). This licence applies to the analysis code only; the underlying patient data are not included and are not covered by it.
