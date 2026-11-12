# Acquire the original MVTec AD

1. Open the [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
   and use its dataset download form. Choose the complete original MVTec AD,
   not MVTec AD 2 or LOCO AD. Download all 15 categories.
2. Review and retain the CC BY-NC-SA 4.0 license and attribution. Cite Bergmann
   et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly
   Detection*, CVPR 2019, and the extended IJCV paper linked on the official page.
3. Put the archive inside this project's ignored `data/` directory. Record source
   URL, retrieval UTC timestamp, filename, bytes and `Get-FileHash -Algorithm SHA256`
   output in `data/acquisition.json`. Compare any official checksum if provided;
   otherwise label the checksum self-computed, not publisher-authenticated.
4. Extract so `data/mvtec_ad/bottle/train/good/*.png` exists, with the same layout
   for every category and `test/` plus `ground_truth/` beside `train/`. Remove
   accidental extra wrapper directories by selecting the correct extraction root.
   Never source files from another local project or symlink to another project.
5. Run `python scripts/execute_eda.py`. Missing categories, malformed images,
   incorrect masks, duplicates or split leakage must be resolved before training.
   Full manifests and rendered image panels remain ignored in `outputs/`.

Expected categories (names, not locally measured counts): bottle, cable, capsule,
carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush,
transistor, wood, zipper. Textures: carpet, grid, leather, tile, wood; the other
ten are objects. A normal image has no official mask file; all-zero evaluation
masks mean normal and must never be described as annotated synthetic anomalies.

Current initialization: no official image, mask or archive was present in this
repository. No category counts or observed benchmark statistics can be reported.
