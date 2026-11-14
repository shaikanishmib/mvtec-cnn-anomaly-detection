# Visual review of retained seed-2026 cases

The corrected nine-row panel was inspected after confirmation, using only stored
maps and original annotations. It displays strongest false positives, weakest
false negatives and successful controls for bottle, cable and screw. Bottle has
no seed-2026 false negative, so its control is the lowest-scoring true normal.
The panel is local at `outputs/padim/failure-panels.png`; the executed notebook
with these panels is `outputs/padim/05_padim.review.executed.ipynb`. Public notebook
panel output is cleared. No benchmark inference or threshold was changed.

- Bottle: the false normal alarm is concentrated near the bottle interior; the
  low-scoring normal has no threshold overlay despite a visually bright heatmap.
  The contamination true positive lights the bottom foreign object, with some
  extra unannotated response. Color scales are per image, not comparable scores.
- Cable: the false positive responds on ordinary wire interiors. The missing-wire
  false negative has a negative image margin and no pixel-threshold overlay.
  The bent-wire true positive has a broad overlay over exposed, displaced wires,
  extending beyond parts of the annotation.
- Screw: the false positive has a strong border response. The scratch-neck false
  negative does not cross the threshold despite broad activation elsewhere on
  the screw. The manipulated-front true positive responds near the annotated
  damaged tip, with extra responses near the head/background border.

These selected examples support the distinction between ranking, localization
and calibration transfer. They do not establish that borders, spatial shifts or
smoothing caused the aggregate errors. The full category/seed predictions and
retrospective mask-size strata are the quantitative evidence, not this selected
qualitative panel. The earlier FP-only analysis execution and image are retained
locally as `05_padim.full.executed.ipynb` and `failure-panels-v1.png`.
