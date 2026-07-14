# CAFT-PCA: top-5 PCs per layer (variance-selected)

- source: `output/caft_pca/20260714_005935_full/caft_pcs_completions.pt` (pca_source=completions)
- layers: [9, 24, 38]
- total ablated directions: **15**

| Layer | PCs ablated | top-5 explained variance |
|-------|-------------|--------------------------|
| 9 | [0, 1, 2, 3, 4] | [6.4, 0.52, 0.39, 0.27, 0.2] |
| 24 | [0, 1, 2, 3, 4] | [72.07, 32.16, 24.26, 10.9, 8.01] |
| 38 | [0, 1, 2, 3, 4] | [1207.93, 1004.91, 719.33, 493.99, 387.26] |
