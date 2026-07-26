# Shared helpers for every script in this repo. Dot-source it:
#     . "$PSScriptRoot\_common.ps1"
#
# Windows PowerShell 5.1 compatible: no '&&', no ternaries, no null-coalescing.

$ErrorActionPreference = 'Stop'

$script:RepoRoot   = Split-Path -Parent $PSScriptRoot
$script:VenvDir    = Join-Path $RepoRoot '.venv'
$script:VendorDir  = Join-Path $RepoRoot 'vendor'
$script:LlamaDir   = Join-Path $VendorDir 'llamacpp'
$script:ModelsDir  = Join-Path $RepoRoot 'models'

function Write-Step   { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Ok     { param($m) Write-Host "    OK   $m" -ForegroundColor Green }
function Write-Warn2  { param($m) Write-Host "    WARN $m" -ForegroundColor Yellow }
function Write-Info   { param($m) Write-Host "    $m" -ForegroundColor DarkGray }

function Get-FreeSpaceGB {
    param([string]$Path = $script:RepoRoot)
    $qualifier = (Split-Path -Qualifier (Resolve-Path $Path))
    $drive = Get-PSDrive ($qualifier -replace ':', '')
    return [math]::Round($drive.Free / 1GB, 2)
}

function Get-VenvPython {
    <#  Path to the venv interpreter, or $null if the venv isn't built yet. #>
    $p = Join-Path $script:VenvDir 'Scripts\python.exe'
    if (Test-Path $p) { return $p }
    return $null
}

function Get-AnyPython {
    <#  Prefer the venv interpreter; fall back to system python.
        config.py is stdlib-only, so either can read the config. #>
    $venv = Get-VenvPython
    if ($venv) { return $venv }
    $sys = Get-Command python -ErrorAction SilentlyContinue
    if ($sys) { return $sys.Source }
    throw "No Python found. Install Python 3.10+ or run scripts\setup_local.ps1 first."
}

function Get-TinyllmConfig {
    <#  Read src/tinyllm/config.py and return it as a PSCustomObject.

        Scripts pull ports, filenames, quant levels and the llama.cpp build tag
        from here rather than hardcoding them, so config.py stays the single
        source of truth across Python, PowerShell and the notebooks. #>
    $py = Get-AnyPython
    $src = Join-Path $script:RepoRoot 'src'
    $code = @"
import json, sys
sys.path.insert(0, r'$src')
from tinyllm import config
print(json.dumps(config.as_dict()))
"@
    $json = & $py -c $code
    if ($LASTEXITCODE -ne 0) { throw "Failed to read config.py (exit $LASTEXITCODE)" }
    return ($json | ConvertFrom-Json)
}

function Get-LlamaBinDir {
    <#  Directory holding llama-cli.exe. The layout inside llama.cpp's release
        zip has moved between builds, so locate it rather than assume it. #>
    if (-not (Test-Path $script:LlamaDir)) {
        throw "llama.cpp not found at $($script:LlamaDir). Run scripts\setup_local.ps1 first."
    }
    $cli = Get-ChildItem -Path $script:LlamaDir -Filter 'llama-cli.exe' -Recurse -ErrorAction SilentlyContinue |
           Select-Object -First 1
    if (-not $cli) {
        throw "llama-cli.exe not found under $($script:LlamaDir). Delete vendor\ and re-run setup_local.ps1."
    }
    return $cli.DirectoryName
}

function Get-LlamaTool {
    <#  Full path to a llama.cpp executable, e.g. Get-LlamaTool 'llama-server' #>
    param([Parameter(Mandatory)][string]$Name)
    $exe = Join-Path (Get-LlamaBinDir) "$Name.exe"
    if (-not (Test-Path $exe)) { throw "$Name.exe not found in $(Get-LlamaBinDir)" }
    return $exe
}

function Get-ModelPath {
    <#  Resolve a GGUF in models/ by quant level, e.g. Get-ModelPath 'Q8_0'.
        Pass 'f16' for the unquantized conversion. #>
    param([Parameter(Mandatory)][string]$Quant)
    $cfg = Get-TinyllmConfig
    $name = "$($cfg.project)-$Quant.gguf"
    $path = Join-Path $script:ModelsDir $name
    if (-not (Test-Path $path)) {
        throw "$name not found in models\. Run pull_model.ps1 (f16) then quantize.ps1."
    }
    return $path
}
