# Local development helper for the PostgreSQL-canonical architecture (Windows).
#
# PowerShell has no `source .env`, so every command loads .env.local into this
# process, pins ENGRAM_CONFIG_PATH to config.local.yaml, and runs one step.
#
# Usage (from the repo root):
#   .\scripts\local.ps1 infra-up      # start PostgreSQL, Temporal, Neo4j, Redis in Docker
#   .\scripts\local.ps1 infra-ps      # show container status
#   .\scripts\local.ps1 infra-down    # stop containers (data volumes are kept)
#   .\scripts\local.ps1 migrate       # apply PostgreSQL schema migrations
#   .\scripts\local.ps1 init          # check PostgreSQL/Neo4j/Redis, create Neo4j indexes
#   .\scripts\local.ps1 health        # print component readiness
#   .\scripts\local.ps1 api           # API on 127.0.0.1:8001        (own terminal)
#   .\scripts\local.ps1 worker        # Temporal worker, all queues  (own terminal)
#   .\scripts\local.ps1 dispatcher    # outbox -> Temporal dispatcher (own terminal)
#   .\scripts\local.ps1 smoke         # end-to-end smoke test (needs api/worker/dispatcher running)
#   .\scripts\local.ps1 quota         # OpenCode quota (shared workspace)
#   .\scripts\local.ps1 judge-test    # judge self-test, expect 4/4
#   .\scripts\local.ps1 bench <args>  # run_locomo.py, e.g. bench --start-conv 0 --limit-convs 1
#   .\scripts\local.ps1 merge <dirs>  # merge_results.py over run directories

#
# Overrides (environment variables): ENGRAM_LOCAL_ENV_FILE, ENGRAM_LOCAL_PYTHON.

# A plain (non-advanced) param block with a single parameter, so every extra
# word lands in $args and is passed through to the Python script unchanged.
param(
    [ValidateSet("infra-up", "infra-ps", "infra-down", "migrate", "init", "health", "api",
        "worker", "dispatcher", "smoke", "quota", "judge-test", "bench", "merge")]
    [string]$Command
)
$EnvFile = $env:ENGRAM_LOCAL_ENV_FILE
$Python = $env:ENGRAM_LOCAL_PYTHON

$ErrorActionPreference = "Stop"
$Rest = @($args)  # extra words for bench/merge (captured before any nested block)
if (-not $Command) { Write-Error "usage: .\scripts\local.ps1 <command> [args]  (see the header)" }

# Repo root = parent of this script's directory.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not $EnvFile) { $EnvFile = Join-Path $RepoRoot ".env.local" }
if (-not (Test-Path $EnvFile)) { Write-Error "env file not found: $EnvFile" }

Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
        $i = $line.IndexOf('=')
        Set-Item -Path ("Env:" + $line.Substring(0, $i).Trim()) -Value $line.Substring($i + 1).Trim()
    }
}
$env:ENGRAM_CONFIG_PATH = Join-Path $RepoRoot "config.local.yaml"
$env:ENGRAM_LOG_FORMAT = "plain"
# The embedding model is already in the local Hugging Face cache; skip the
# per-start online update check (the production Dockerfile does the same).
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# Default interpreter: Code\.engram_new (kept separate from the old repo's
# Code\.engram, which has the old `engram` package installed under the same name).
if (-not $Python) {
    $Python = Join-Path (Split-Path -Parent (Split-Path -Parent $RepoRoot)) ".engram_new\Scripts\python.exe"
}

$Compose = @("compose", "--env-file", $EnvFile, "-f", "docker-compose.local.yml")

switch ($Command) {
    "infra-up"   { docker @Compose up -d; docker @Compose ps -a }
    "infra-ps"   { docker @Compose ps -a }
    "infra-down" { docker @Compose down }
    "migrate"    { & $Python -m engram.cli migrate }
    "init"       { & $Python -m engram.cli init }
    "health"     { & $Python -m engram.cli health }
    "api"        { & $Python -m uvicorn engram.api.app:app --host 127.0.0.1 --port 8001 }
    "worker"     { & $Python -m engram.temporal.worker --queue all }
    "dispatcher" { & $Python -m engram.temporal.dispatcher }
    "smoke"      { & $Python benchmarks\smoke_canonical.py }
    "quota"      { & $Python benchmarks\check_quota.py }
    "judge-test" { & $Python benchmarks\judge.py }
    "bench"      { & $Python benchmarks\run_locomo.py @Rest }
    "merge"      { & $Python benchmarks\merge_results.py @Rest }
}
exit $LASTEXITCODE
