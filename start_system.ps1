# ===========================================================
# start_system.ps1
# SIH Problem Statement ID: 26127 (BEL)
# One-Click ANPR System Startup Script (PowerShell)
#
# Starts:
#   1. Python ANPR Inference Server  -> http://127.0.0.1:5001
#   2. Node.js Dashboard Server      -> https://127.0.0.1:3000
#
# Usage:
#   Right-click -> Run with PowerShell
#   OR: powershell -ExecutionPolicy Bypass -File start_system.ps1
# ===========================================================

$Host.UI.RawUI.WindowTitle = "SIH-26127 ANPR System Startup"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SIH Problem Statement ID: 26127 (BEL)" -ForegroundColor Cyan
Write-Host "  City-Wide Indian ANPR Intelligence System" -ForegroundColor Cyan
Write-Host "  One-Click System Startup (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# --- Helper: Test TCP port is listening ---
function Test-Port {
    param([string]$TargetHost, [int]$Port, [int]$TimeoutMs = 500)
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $ar = $tcp.BeginConnect($TargetHost, $Port, $null, $null)
        $ok = $ar.AsyncWaitHandle.WaitOne($TimeoutMs)
        if ($ok) { $tcp.EndConnect($ar); $tcp.Close(); return $true }
        $tcp.Close(); return $false
    } catch { return $false }
}

# --- Step 1: Check Python ---
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
try {
    $pyVer = python --version 2>&1
    Write-Host "      $pyVer" -ForegroundColor Green
    Write-Host "  [OK] Python found." -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Python not found in PATH. Please install Python 3.9+ and add to PATH." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# --- Step 2: Check Node.js ---
Write-Host "[2/5] Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVer = node --version 2>&1
    Write-Host "      $nodeVer" -ForegroundColor Green
    Write-Host "  [OK] Node.js found." -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Node.js not found in PATH. Please install Node.js 18+ and add to PATH." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# --- Step 3: npm install if needed ---
Write-Host "[3/5] Checking Node dependencies..." -ForegroundColor Yellow
$nodeModulesPath = Join-Path $ScriptDir "node_modules"
if (-not (Test-Path $nodeModulesPath)) {
    Write-Host "  [INFO] node_modules not found. Running npm install..." -ForegroundColor Yellow
    Push-Location $ScriptDir
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] npm install failed." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Pop-Location
}
Write-Host "  [OK] Node dependencies ready." -ForegroundColor Green
Write-Host ""

# --- Step 4: Check if ANPR server already running ---
Write-Host "[4/5] Starting Python ANPR Inference Server (port 5001)..." -ForegroundColor Yellow
if (Test-Port "127.0.0.1" 5001) {
    Write-Host "  [INFO] Port 5001 already in use. ANPR server appears to be running." -ForegroundColor Green
} else {
    $anprServerScript = Join-Path $ScriptDir "scripts\anpr_server.py"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ScriptDir'; python '$anprServerScript'" -WindowStyle Normal
    Write-Host "  [INFO] ANPR server window launched. Waiting 10 seconds for model load..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10

    # Verify startup
    $retries = 3
    $started = $false
    for ($i = 0; $i -lt $retries; $i++) {
        if (Test-Port "127.0.0.1" 5001) {
            $started = $true
            break
        }
        Write-Host "  [INFO] Still waiting for ANPR server... ($($i+1)/$retries)" -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    }
    if ($started) {
        Write-Host "  [OK] ANPR server is responding at http://127.0.0.1:5001" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] ANPR server may still be loading (first-run model init can take 20-30s)." -ForegroundColor Yellow
        Write-Host "         Check the ANPR server window for details." -ForegroundColor Yellow
    }
}
Write-Host ""

# --- Step 5: Start Node.js dashboard ---
Write-Host "[5/5] Starting Node.js Dashboard Server (port 3000)..." -ForegroundColor Yellow
if (Test-Port "127.0.0.1" 3000) {
    Write-Host "  [INFO] Port 3000 already in use. Dashboard server appears to be running." -ForegroundColor Green
} else {
    $serverScript = Join-Path $ScriptDir "server.js"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ScriptDir'; node '$serverScript'" -WindowStyle Normal
    Write-Host "  [INFO] Node.js dashboard window launched. Waiting 3 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3

    if (Test-Port "127.0.0.1" 3000) {
        Write-Host "  [OK] Dashboard server is responding at https://127.0.0.1:3000" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Dashboard server may still be starting. Check the Node.js window." -ForegroundColor Yellow
    }
}
Write-Host ""

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SYSTEM STARTUP COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  ANPR Inference API:  http://127.0.0.1:5001" -ForegroundColor White
Write-Host "  Dashboard UI:        https://127.0.0.1:3000" -ForegroundColor White
Write-Host "  Health Check:        http://127.0.0.1:5001/health" -ForegroundColor White
Write-Host "  Config API:          http://127.0.0.1:5001/config" -ForegroundColor White
Write-Host "  Diagnostics Panel:   https://127.0.0.1:3000/dashboard.html" -ForegroundColor White
Write-Host ""
Write-Host "  Run diagnostics benchmark:" -ForegroundColor DarkCyan
Write-Host "    python scripts\diagnose_pipeline.py" -ForegroundColor DarkCyan
Write-Host ""
Write-Host "  NOTE: Browser may warn about the self-signed HTTPS certificate." -ForegroundColor DarkYellow
Write-Host "        Click 'Advanced' -> 'Proceed' to continue." -ForegroundColor DarkYellow
Write-Host ""

$open = Read-Host "Open dashboard in browser? (Y/n)"
if ($open -ne "n" -and $open -ne "N") {
    Start-Process "https://127.0.0.1:3000/dashboard.html"
}

Write-Host ""
Write-Host "  Press Enter to close this startup window (servers remain running)."
Read-Host