<#
.SYNOPSIS
    Writes your GitHub repo URL into the bootstrap cell of every Colab notebook.

.DESCRIPTION
    Each Colab notebook starts by cloning this repository, so each one needs the
    URL. Editing eight files by hand is tedious and easy to get half-right -- a
    notebook still pointing at CHANGEME fails several cells in with a confusing
    error.

    With no arguments this reads the URL from your 'origin' remote, which is the
    normal case: set the remote once, run this, done.

    The actual work is in scripts/set_repo_url.py. It lives in a real file rather
    than an embedded here-string because PowerShell strips bare double quotes
    when passing arguments to a native exe, which corrupts any Python source
    containing string literals.

.PARAMETER Url
    Repo URL to use. Defaults to the 'origin' remote. Accepts SSH or https.

.EXAMPLE
    git remote add origin https://github.com/pythonstudentiam/e2e_llm_demo.git
    .\scripts\set_repo_url.ps1

.EXAMPLE
    .\scripts\set_repo_url.ps1 -Url https://github.com/pythonstudentiam/e2e_llm_demo.git
#>
[CmdletBinding()]
param([string]$Url)

. "$PSScriptRoot\_common.ps1"

if (-not $Url) {
    # $ErrorActionPreference is 'Stop', and a missing remote makes git write to
    # stderr -- which would surface git's terse message instead of the useful one
    # below. Let cmd swallow it so we control what the user sees.
    $Url = & cmd /c "git -C `"$RepoRoot`" remote get-url origin 2>NUL"

    if (-not $Url) {
        throw @"
No 'origin' remote configured, and no -Url given.

  1. Create an empty repo at https://github.com/new
     (no README, no .gitignore, no license -- this repo already has them)

  2. Then run:
       git remote add origin https://github.com/pythonstudentiam/<repo-name>.git
       .\scripts\set_repo_url.ps1
"@
    }
    Write-Info "using origin remote"
}

Write-Step "Setting REPO_URL in Colab notebooks"

$py = Get-AnyPython
& $py (Join-Path $PSScriptRoot 'set_repo_url.py') $Url
if ($LASTEXITCODE -ne 0) { throw "Failed to update notebooks (exit $LASTEXITCODE)" }

Write-Step "Next"
Write-Host "    git add -A"
Write-Host "    git commit -m `"Set Colab repo URL`""
Write-Host "    git push -u origin main"
Write-Host ""
Write-Host "  Then open notebooks\colab\01_data.ipynb in Colab." -ForegroundColor White
Write-Host ""

exit 0
