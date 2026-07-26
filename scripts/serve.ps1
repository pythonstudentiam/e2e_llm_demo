<#
.SYNOPSIS
    Starts llama-server: an OpenAI-compatible HTTP API for your model.

.DESCRIPTION
    llama-server exposes /v1/chat/completions and /v1/completions, so anything
    that speaks the OpenAI API -- the openai SDK, Continue.dev, curl -- can talk
    to your model with only a base-URL change. It also serves a browser UI at
    the root URL.

    The chat template travels inside the GGUF, so the server knows how to format
    turns without being told. That is the last of the four hops for the template
    string defined back in config.py.

.PARAMETER Quant
    Which quantization to serve. Defaults to ServeConfig.default_quant (Q8_0).

.PARAMETER Port
    Override the port. Defaults to ServeConfig.port (8080).

.PARAMETER Threads
    CPU threads. Defaults to ServeConfig.threads (4 -- this laptop has 2 cores / 4 logical).

.EXAMPLE
    .\scripts\serve.ps1
    .\scripts\serve.ps1 -Quant Q4_K_M
#>
[CmdletBinding()]
param(
    [string]$Quant,
    [int]$Port,
    [int]$Threads
)

. "$PSScriptRoot\_common.ps1"

$cfg = Get-TinyllmConfig
if (-not $Quant)   { $Quant   = $cfg.serve.default_quant }
if (-not $Port)    { $Port    = $cfg.serve.port }
if (-not $Threads) { $Threads = $cfg.serve.threads }

$model = Get-ModelPath $Quant
$server = Get-LlamaTool 'llama-server'
$mb = [math]::Round((Get-Item $model).Length / 1MB, 1)

Write-Step "Starting llama-server"
Write-Info "model    $(Split-Path -Leaf $model) ($mb MB)"
Write-Info "context  $($cfg.serve.ctx_size) tokens"
Write-Info "threads  $Threads"

# Fail early with a clear message rather than letting the server bind-fail.
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    throw "Port $Port is already in use (PID $($busy[0].OwningProcess)). Stop it, or pass -Port <other>."
}

Write-Host ""
Write-Host "  Web UI      http://$($cfg.serve.host):$Port" -ForegroundColor Green
Write-Host "  API base    http://$($cfg.serve.host):$Port/v1" -ForegroundColor Green
Write-Host "  Model id    $($cfg.serve.served_model_name)" -ForegroundColor Green
Write-Host ""
Write-Host "  Try it from another terminal:" -ForegroundColor White
Write-Host "    python clients\chat_cli.py"
Write-Host ""
Write-Host "  Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

# --jinja makes the server use the chat template embedded in the GGUF rather
# than a built-in guess. Without it, a ChatML model can be served with the
# wrong turn format and quietly produce worse output.
& $server `
    --model $model `
    --alias $cfg.serve.served_model_name `
    --ctx-size $cfg.serve.ctx_size `
    --threads $Threads `
    --host $cfg.serve.host `
    --port $Port `
    --jinja
