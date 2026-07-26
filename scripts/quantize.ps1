<#
.SYNOPSIS
    Quantizes the f16 GGUF to Q8_0, Q5_K_M and Q4_K_M.

.DESCRIPTION
    Quantization stores weights in fewer bits. The K-quants are not uniform:
    they split each block of weights into sub-blocks with their own scale, and
    they assign more bits to the tensors that matter most (attention output and
    the feed-forward down-projection), which is why Q4_K_M beats a flat 4-bit
    scheme at the same size.

      Q8_0     8-bit, one scale per 32 weights. Near-lossless, ~53% of f16.
      Q5_K_M   5-bit K-quant, mixed precision. ~35% of f16.
      Q4_K_M   4-bit K-quant, mixed precision. ~28% of f16.

    A caution specific to this project: quantization guidance is written for
    7B+ models. A 15.7M-parameter model has far less parameter redundancy to
    absorb rounding error, so expect Q4_K_M to hurt noticeably here.
    notebooks/local/10_quant_tradeoff.ipynb measures it rather than guessing.

.PARAMETER Levels
    Override the quantization levels. Defaults to QuantConfig.levels.

.PARAMETER Force
    Re-quantize even if the output already exists.
#>
[CmdletBinding()]
param(
    [string[]]$Levels,
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$cfg = Get-TinyllmConfig
if (-not $Levels) { $Levels = @($cfg.quant.levels) }

$src = Join-Path $ModelsDir "$($cfg.project)-f16.gguf"
if (-not (Test-Path $src)) {
    throw "$src not found. Run scripts\pull_model.ps1 first."
}

$quantize = Get-LlamaTool 'llama-quantize'
$srcMB = [math]::Round((Get-Item $src).Length / 1MB, 2)

Write-Step "Quantizing $($cfg.project)-f16.gguf ($srcMB MB)"
Write-Info "levels: $($Levels -join ', ')"

$results = @()
$results += [pscustomobject]@{ Quant = 'f16'; SizeMB = $srcMB; PctOfF16 = 100.0 }

foreach ($level in $Levels) {
    $dest = Join-Path $ModelsDir "$($cfg.project)-$level.gguf"

    if ((Test-Path $dest) -and (-not $Force)) {
        Write-Info "$level already exists, skipping (use -Force to redo)"
    } else {
        Write-Info "producing $level ..."
        # llama-quantize writes its progress log to stderr; let cmd merge the
        # streams so PowerShell 5.1 does not wrap each line in an ErrorRecord.
        $out = @(& cmd /c "`"$quantize`" `"$src`" `"$dest`" $level 2>&1")
        if ($LASTEXITCODE -ne 0) {
            $out | Select-Object -Last 15 | ForEach-Object { Write-Host "      $_" }
            throw "llama-quantize failed for $level (exit $LASTEXITCODE)"
        }
    }

    $mb = [math]::Round((Get-Item $dest).Length / 1MB, 2)
    $results += [pscustomobject]@{
        Quant    = $level
        SizeMB   = $mb
        PctOfF16 = [math]::Round(100 * $mb / $srcMB, 1)
    }
    Write-Ok "$level -> $mb MB"
}

Write-Step "Size comparison"
$results | Format-Table -AutoSize

$free = Get-FreeSpaceGB
Write-Info "free disk: $free GB"

Write-Step "Next"
Write-Host "    .\scripts\serve.ps1"
Write-Host "    Then measure what quantization actually cost:"
Write-Host "    notebooks\local\10_quant_tradeoff.ipynb"
Write-Host ""

exit 0
