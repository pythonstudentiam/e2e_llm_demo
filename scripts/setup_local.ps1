<#
.SYNOPSIS
    Sets up the LOCAL tier: a torch-free Python venv plus llama.cpp binaries.

.DESCRIPTION
    This machine never trains anything and never touches PyTorch. It quantizes,
    serves, and analyses. That keeps the footprint around 450-700 MB, which is
    what makes this project possible on ~2 GB of free disk.

    llama.cpp is downloaded as a prebuilt release, not built from source --
    building would require cmake plus MSVC Build Tools (several GB), whereas the
    CPU release zip is 17 MB.

.PARAMETER SkipVenv
    Leave the Python environment alone; only fetch llama.cpp.

.PARAMETER SkipLlama
    Only build the Python environment.

.PARAMETER NoNotebook
    Skip Jupyter (~200 MB). Use if disk is very tight; the local notebooks
    won't run, but quantizing, serving and the CLI client still will.

.PARAMETER Force
    Re-download and re-extract llama.cpp even if it's already present.

.EXAMPLE
    .\scripts\setup_local.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipVenv,
    [switch]$SkipLlama,
    [switch]$NoNotebook,
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$startFree = Get-FreeSpaceGB
Write-Host ""
Write-Host "  tinyllm - local tier setup" -ForegroundColor White
Write-Host "  ==========================" -ForegroundColor White
Write-Info "repo:       $RepoRoot"
Write-Info "free disk:  $startFree GB"

if ($startFree -lt 1.0) {
    Write-Warn2 "Under 1 GB free. This will likely fail partway through."
    Write-Warn2 "Free some space first, or re-run with -NoNotebook to save ~200 MB."
} elseif ($startFree -lt 1.8) {
    Write-Warn2 "Disk is tight ($startFree GB). Expect to use ~450-700 MB here."
}

# ---------------------------------------------------------------------------
# 1. Python environment (deliberately no torch -- see pyproject.toml)
# ---------------------------------------------------------------------------
if (-not $SkipVenv) {
    Write-Step "Creating Python environment"

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv not found on PATH. Install from https://docs.astral.sh/uv/ then re-run."
    }

    Push-Location $RepoRoot
    try {
        if (-not (Test-Path $VenvDir)) {
            uv venv
            if ($LASTEXITCODE -ne 0) { throw "uv venv failed (exit $LASTEXITCODE)" }
            Write-Ok "venv created at .venv"
        } else {
            Write-Info ".venv already exists, reusing"
        }

        $py = Get-VenvPython
        if (-not $py) { throw "venv python missing after creation" }

        if ($NoNotebook) { $target = '-e', '.' } else { $target = '-e', '.[notebook]' }

        Write-Info "installing dependencies (no torch by design)..."
        uv pip install --python $py @target
        if ($LASTEXITCODE -ne 0) { throw "dependency install failed (exit $LASTEXITCODE)" }
        Write-Ok "dependencies installed"

        # Guard the core design constraint. If torch ever sneaks in as a
        # transitive dep it will silently eat the remaining disk.
        & $py -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') is None else 1)"
        if ($LASTEXITCODE -ne 0) {
            Write-Warn2 "torch got installed into the local venv -- that should not happen."
            Write-Warn2 "Check pyproject.toml; the local tier is meant to stay torch-free."
        } else {
            Write-Ok "confirmed torch-free"
        }

        if (-not $NoNotebook) {
            & $py -m ipykernel install --user --name tinyllm --display-name "tinyllm (local, no torch)" | Out-Null
            if ($LASTEXITCODE -eq 0) { Write-Ok "Jupyter kernel 'tinyllm (local, no torch)' registered" }
        }
    } finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------------------
# 2. llama.cpp prebuilt binaries
# ---------------------------------------------------------------------------
if (-not $SkipLlama) {
    Write-Step "Fetching llama.cpp"

    $cfg    = Get-TinyllmConfig
    $build  = $cfg.serve.llamacpp_build
    $asset  = $cfg.serve.llamacpp_asset
    $url    = "https://github.com/ggml-org/llama.cpp/releases/download/$build/$asset"

    $already = $false
    if ((Test-Path $LlamaDir) -and (-not $Force)) {
        $probe = Get-ChildItem -Path $LlamaDir -Filter 'llama-cli.exe' -Recurse -ErrorAction SilentlyContinue |
                 Select-Object -First 1
        if ($probe) { $already = $true }
    }

    if ($already) {
        Write-Info "already present at vendor\llamacpp (use -Force to refresh)"
    } else {
        New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null
        $zip = Join-Path $VendorDir $asset

        Write-Info "downloading $asset (~17 MB)"
        Write-Info $url
        $prevProgress = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'   # ~10x faster Invoke-WebRequest
        try {
            Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        } finally {
            $ProgressPreference = $prevProgress
        }
        $sizeMB = [math]::Round((Get-Item $zip).Length / 1MB, 1)
        Write-Ok "downloaded ($sizeMB MB)"

        if (Test-Path $LlamaDir) { Remove-Item -Recurse -Force $LlamaDir }
        Expand-Archive -Path $zip -DestinationPath $LlamaDir -Force
        Remove-Item $zip -Force
        Write-Ok "extracted to vendor\llamacpp"
    }

    # Verify the toolchain we actually depend on downstream.
    $binDir = Get-LlamaBinDir
    Write-Info "binaries in: $binDir"

    $needed = @('llama-cli', 'llama-server', 'llama-quantize', 'llama-perplexity')
    $missing = @()
    foreach ($n in $needed) {
        if (Test-Path (Join-Path $binDir "$n.exe")) { Write-Ok "$n.exe" }
        else { $missing += $n; Write-Warn2 "$n.exe MISSING" }
    }
    if ($missing.Count -gt 0) {
        throw "Missing llama.cpp tools: $($missing -join ', '). Wrong asset for this platform?"
    }

    # llama-cli prints its banner to stderr. In Windows PowerShell 5.1, using
    # '2>&1' on a native exe wraps each stderr line in an ErrorRecord and trips
    # $ErrorActionPreference='Stop'. Let cmd.exe do the merge instead.
    # Capture fully before slicing: piping a native command straight into
    # 'Select-Object -First' stops the pipeline early, kills the process, and
    # leaves $LASTEXITCODE at 255 even though the command itself succeeded.
    $cliExe = Join-Path $binDir 'llama-cli.exe'
    $verAll = @(& cmd /c "`"$cliExe`" --version 2>&1")
    $ver = $verAll | Select-Object -First 2
    Write-Info "version: $($ver -join ' | ')"
}

# ---------------------------------------------------------------------------
# 3. Report
# ---------------------------------------------------------------------------
$endFree = Get-FreeSpaceGB
$used = [math]::Round($startFree - $endFree, 2)

Write-Step "Done"
Write-Info "disk used by setup: $used GB"
Write-Info "free remaining:     $endFree GB"

Write-Host ""
Write-Host "  Next:" -ForegroundColor White
Write-Host "    1. Set HubConfig.user in src\tinyllm\config.py to your HF username"
Write-Host "    2. git init, commit, push this repo to GitHub"
Write-Host "    3. Open notebooks\colab\01_data.ipynb in Colab and work through 01 -> 08"
Write-Host "    4. Come back and run: .\scripts\pull_model.ps1"
Write-Host ""
Write-Host "  Sanity check the config any time with:" -ForegroundColor White
Write-Host "    python src\tinyllm\config.py"
Write-Host ""

# Don't let a stray $LASTEXITCODE from a native tool become the script's status.
exit 0
