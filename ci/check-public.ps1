# Asset-free source check. No host Python, dataset, weights or GPU required.
[CmdletBinding()]
param([ValidatePattern('^[a-z0-9-]+$')][string]$RunId = 'public')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$sourceRoot = Join-Path $projectRoot "outputs/portfolio/$RunId-source"
$resultRoot = Join-Path $projectRoot "outputs/portfolio/$RunId-results"
if (Test-Path -LiteralPath $sourceRoot) {
    throw 'Public-source snapshot already exists. Preserve its evidence and choose a fresh checkout for another independent run.'
}
New-Item -ItemType Directory -Force -Path $sourceRoot,$resultRoot | Out-Null
# Tracked public files only; overlay explicit portfolio additions before their
# first commit. Never enumerate/copy data, banks, credentials or private caches.
$publicNames = @(& git ls-files)
if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed' }
$additions = @('ci','.github/workflows','reports/portfolio','docs/index.md','docs/project_overview.md',
    'docs/reproducibility.md','docs/licensing_and_attribution.md','SECURITY.md',
    'scripts/check_portfolio.py','scripts/run_public_tests.py','tests/test_portfolio.py')
foreach ($name in $additions) {
    $target = Join-Path $projectRoot $name
    if (Test-Path -LiteralPath $target -PathType Leaf) { $publicNames += $name }
    elseif (Test-Path -LiteralPath $target -PathType Container) {
        foreach ($item in Get-ChildItem -LiteralPath $target -File -Recurse) {
            if ($item.FullName -notmatch '[\\/]__pycache__[\\/]') {
                $publicNames += $item.FullName.Substring($projectRoot.Length + 1).Replace('\','/')
            }
        }
    }
}
foreach ($name in ($publicNames | Sort-Object -Unique)) {
    if ($name -match '(^|/)(data|checkpoints|node_modules|outputs|\.git|\.venv)(/|$)' -or $name -match '\.(pt|pth|npy|npz|pem|key)$') {
        throw "Unexpected private asset in public source list: $name"
    }
    $destination = Join-Path $sourceRoot $name
    New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
    Copy-Item -LiteralPath (Join-Path $projectRoot $name) -Destination $destination
}
$baseImages = @(& docker image ls --quiet fieldwork-backend:local)
if ($LASTEXITCODE -ne 0) { throw 'Docker engine is unavailable' }
if ($baseImages.Count -eq 0) {
    & docker compose -p fieldwork-public --env-file deployment/.env.example build backend
    if ($LASTEXITCODE -ne 0) { throw 'CPU source-check base build failed' }
}
$command = 'python -m pip install --no-deps --target /tmp/ci-deps -r ci/requirements.txt && python -m pip check && python scripts/check_portfolio.py --require-asset-free --output /results/checks.json && python scripts/run_public_tests.py --junitxml /results/pytest.xml --basetemp /tmp/pytest-public --cache-dir /tmp/pytest-cache'
& docker run --rm --cap-drop ALL --security-opt no-new-privileges --cpus 2 --memory 3g `
    --mount "type=bind,source=$sourceRoot,target=/workspace,readonly" `
    --mount "type=bind,source=$resultRoot,target=/results" `
    -e PYTHONPATH=/tmp/ci-deps:/workspace/src:/workspace -e CUDA_VISIBLE_DEVICES= `
    -e ANOMALY_ARTIFACT_ROOT=/workspace -e ANOMALY_MANIFEST=/workspace/configs/patchcore_deployment_v1.json -e ANOMALY_DEVICE=cpu `
    -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -w /workspace --entrypoint sh fieldwork-backend:local -c $command
if ($LASTEXITCODE -ne 0) { throw 'Asset-free checks failed; retained snapshot and results are under outputs/portfolio' }
Write-Output 'Fresh public-source checks passed; no private artifact directory or GPU was mounted.'
