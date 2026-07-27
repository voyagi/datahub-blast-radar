# Blast Radar demo driver - run this while recording, one take, no typing.
#
# Usage:
#   pwsh -NoProfile -File events\demo-run.ps1
#
# Press Enter between steps so narration can catch up. Nothing here is
# destructive: the publish step writes tags, scores and one brief into the
# LOCAL DataHub only, and re-running updates the same brief rather than
# stacking up new ones.

$ErrorActionPreference = 'Continue'
$repo = Join-Path $env:USERPROFILE "projects\datahub-blast-radar"
$asset = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.ORDER_ENTRY_DB.analytics.order_details,PROD)"

$env:DATAHUB_GMS_URL = "http://localhost:8080"
$env:TOOLS_IS_MUTATION_ENABLED = "true"

function Step {
    param([string]$Title)
    Write-Host ""
    Write-Host ("  " + $Title) -ForegroundColor Cyan
    Write-Host ""
    Read-Host "  [Enter to run]" | Out-Null
    Clear-Host
}

Clear-Host

# Preflight, before the camera rolls. A cold uvx download mid-demo is 30 dead
# seconds, and an unhealthy GMS is a wasted take.
Write-Host "Preflight..." -ForegroundColor DarkGray
try {
    $health = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing -TimeoutSec 8
    Write-Host ("  GMS: " + $health.StatusCode) -ForegroundColor Green
} catch {
    Write-Host "  GMS is down. Run: datahub docker quickstart --version v1.6.0" -ForegroundColor Red
    exit 1
}
Push-Location $repo
uv run blast-radar doctor | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  doctor failed - fix the connection before recording." -ForegroundColor Red
    Pop-Location
    exit 1
}
Write-Host "  doctor: OK (MCP server cached, tools present)" -ForegroundColor Green
Write-Host ""
Read-Host "  Preflight passed. [Enter to start the demo]" | Out-Null
Clear-Host

Step "1/3  The scan. Thirty-five things are downstream of this model."
uv run blast-radar scan $asset --change "dropping column promo_code" --limit 12

Step "2/3  Publish the verdict back into DataHub."
uv run blast-radar scan $asset --change "dropping column promo_code" --limit 1 --publish

Step "3/3  The tests. No DataHub instance, no network."
uv run pytest -q

Write-Host ""
Write-Host "  Done. Switch to the DataHub UI and refresh a tagged chart." -ForegroundColor Cyan
Write-Host "  Suggested: Executive Summary, or the Order Entry Dashboard." -ForegroundColor DarkGray
Pop-Location
