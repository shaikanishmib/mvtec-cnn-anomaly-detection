# Restart audit

Starting HEAD: `5e3b520538e862dd275bded9abad9a0a8163b814`.
Initial `git status --short` contained only untracked `src/anomaly_detection/padim.py`,
`tests/test_padim.py`, `docs/milestone_status.md`, and `.pytest-tmp/`.
Inspection confirmed the checklist belonged solely to the failed Milestone 7.
The scratch directory contained only the old PaDiM serialization fixture and its
pytest current link. These four explicit workspace paths were removed after
checking resolved paths. Git status was empty immediately afterward.
No tracked/shared changes required reverting. Notebooks 00–04 existed; Notebook
05 did not. No PaDiM experiment directory or pilot scripts existed. Recursive
artifact-name inspection found earlier-model pilots, which were preserved.
No dataset/environment/model/report from completed milestones was removed.

The exact `.venv/Scripts/python.exe` imported the existing package successfully;
no editable install, environment creation or dependency reinstall was necessary.
CUDA PyTorch 2.11.0+cu128 and RTX 4060 Laptop GPU were detected; all 15 dataset
category roots were accessible. Environment details are in environment.json.

## Recoverable launch issues

- Initial default pytest run: 62 passed, 36 fixture setup errors caused by sandbox
  access denial to the Windows pytest temporary directory. An approved rerun
  passed 102 tests including the newly added four numerical tests; two cache
  write warnings. Subsequent checks use explicit fresh workspace scratch paths.
- Initial numerical test scratch parent did not exist: three passed, one fixture
  setup error. Created the output parent; all four passed.
- Initial environment-report writer required its parent directory. Created
  `reports/padim`, reran environment capture and pilot successfully. This failed
  before experiment reservation, GPU fitting or data scoring.
- First development launch used a decimal point in its experiment ID; the shared
  registry rejected it before Gaussian fitting. Replaced epsilon ID spelling by
  integer hundredths. No registered result was overwritten or deleted.

The real GPU pilot passed before any development or confirmation. Development
completed all 16 preregistered candidates; CPU/GPU parity and an independent
native metric fixture passed before freeze. Sources were then frozen by hash.
Final benchmark metrics do not influence model/configuration selection.
