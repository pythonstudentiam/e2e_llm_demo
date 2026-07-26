<#
.SYNOPSIS
    Downloads the f16 GGUF you published in stage 8 from the Hugging Face Hub.

.DESCRIPTION
    This is the handoff between tiers. Colab trained and converted; this machine
    only ever sees the inference format, which is why it needs no PyTorch.

.PARAMETER Repo
    Override the repo id. Defaults to HubConfig.model_repo in config.py.

.PARAMETER Force
    Re-download even if the file is already in models\.
#>
[CmdletBinding()]
param(
    [string]$Repo,
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$cfg = Get-TinyllmConfig
if (-not $Repo) { $Repo = $cfg.hub.model_repo }

if ($Repo -like 'CHANGEME/*') {
    throw "HubConfig.user is still 'CHANGEME'. Set it in src\tinyllm\config.py, or pass -Repo <user>/<model>."
}

$fileName = "$($cfg.project)-f16.gguf"
$dest = Join-Path $ModelsDir $fileName

Write-Step "Pulling $fileName from $Repo"

if ((Test-Path $dest) -and (-not $Force)) {
    $mb = [math]::Round((Get-Item $dest).Length / 1MB, 1)
    Write-Info "already present ($mb MB). Use -Force to re-download."
    exit 0
}

$free = Get-FreeSpaceGB
Write-Info "free disk: $free GB"
if ($free -lt 0.5) { Write-Warn2 "Very little disk left; the download may fail." }

New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$py = Get-VenvPython
if (-not $py) { throw "No venv. Run scripts\setup_local.ps1 first." }

# huggingface_hub caches downloads under ~\.cache\huggingface by default, which
# would mean two copies on a disk that cannot spare one. Download into the cache,
# then move the resolved file into models\ and drop the cache entry.
$code = @"
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

src = hf_hub_download(repo_id=r'$Repo', filename=r'$fileName')
dest = Path(r'$dest')
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dest)
print(f'OK {dest} {dest.stat().st_size}')

# Free the cache copy -- this laptop cannot afford to keep both.
try:
    blob = Path(src).resolve()
    cache_root = blob
    for _ in range(6):
        cache_root = cache_root.parent
        if cache_root.name.startswith('models--'):
            shutil.rmtree(cache_root, ignore_errors=True)
            print('cache entry removed')
            break
except Exception as e:
    print(f'(cache cleanup skipped: {e})')
"@

& $py -c $code
if ($LASTEXITCODE -ne 0) { throw "Download failed (exit $LASTEXITCODE)" }

$mb = [math]::Round((Get-Item $dest).Length / 1MB, 1)
Write-Ok "$fileName ($mb MB) -> models\"

Write-Step "Next"
Write-Host "    .\scripts\quantize.ps1     # produce Q8_0 / Q5_K_M / Q4_K_M"
Write-Host "    .\scripts\serve.ps1        # start llama-server"
Write-Host ""

exit 0
