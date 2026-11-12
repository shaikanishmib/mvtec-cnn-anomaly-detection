# Original MVTec AD: local validation and provenance

The user supplied the complete dataset under this project's data/. Milestone 2
**did not download, move, reorganize or duplicate it**. Automatic discovery found
`data/mvtec_ad/`; categories directly under data/ or inside an extra wrapper also
work. Multiple roots, incomplete categories and linked directories are rejected.
Windows .lnk files are not dereferenced.

All 15 categories passed original-layout validation and decoding, including
`train/good`, `test/<type>` and `ground_truth/<defect>`. There are **5,354 images**:
3,629 training normals, 467 test normals and 1,258 defective test images. All
1,258 official defect masks pass binary-value, nonempty-region, dimensions and
pairing checks. No orphan mask or exact duplicate image was found. All images
are PNG; native square geometry ranges from 700x700 to 1024x1024. See
[category counts](../reports/category_counts.csv),
[defect counts](../reports/category_defect_counts.csv),
[integrity/provenance evidence](../reports/dataset_validation.json) and Notebook 00.

Use the original [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad),
not AD 2 or LOCO AD. Cite Bergmann et al., MVTec AD - A Comprehensive Real-World
Dataset for Unsupervised Anomaly Detection, CVPR 2019, and the extended IJCV paper
linked on the official page. Preserve the **CC BY-NC-SA 4.0** license/attribution.
Original images, masks, archives and image panels stay out of Git.

No original archive, publisher checksum or download timestamp was supplied.
Local SHA256 hashes identify validated files and license/readme; they do not
independently authenticate the original download. Unknown provenance fields stay
null. Do not invent retrieval details or download again merely to fill them.
For a different machine, use the official page's download form and record the
source URL, UTC timestamp, archive size and SHA256, comparing any official checksum.

## Frozen normal-only partitions

Seed 2026, sorted relative paths and a category-specific deterministic shuffle
assign floor(15%) each to development/calibration, with the remainder fitting:

| Partition | Images | Role |
|---|---:|---|
| Fitting | 2,551 | Normal-only learning and fitted preprocessing |
| Development | 539 | Normal validation or explicitly labeled synthetic proxies |
| Calibration | 539 | Fix thresholds after configuration selection |
| Official test | 1,725 | Evaluation only; original masks remain separate |

Four compact path/content manifests are published under reports/partitions/.
Their full SHA256 identities and seed are retained in
[dataset validation metadata](../reports/dataset_validation.json). Original image
bytes and the full pixel-statistics inventory stay ignored under data/ and outputs/.
Test-mask references/hashes occur only in the test manifest. .gitattributes fixes
manifest line endings to LF so identities survive Windows checkouts.

`scripts/prepare_data.py` verifies all files and refuses changed frozen assignments.
Notebook 00 independently recomputes files/assignments and compares exact manifest
text. No paths or exact image identities cross partitions. SHA256 does not exclude
near duplicates or shared acquisition groups. Test images/masks were viewed for
descriptive EDA; never claim unseen test labels or tune using them afterward.
