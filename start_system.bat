@echo off
:: ============================================================
:: start_system.bat
:: SIH Problem Statement ID: 26127 (BEL)
:: One-Click ANPR System Startup Script
::
:: Starts:
::   1. Python ANPR Inference Server  -> http://127.0.0.1:5001
::   2. Node.js Dashboard Server      -> https://127.0.0.1:3000
::
:: Usage: Double-click OR run from project root directory.
:: ============================================================

title SIH-26127 ANPR System Startup
color 0A

echo ============================================================
echo   SIH Problem Statement ID: 26127 (BEL)
echo   City-Wide Indian ANPR Intelligence System
echo   One-Click System Startup
echo ============================================================
echo.

:: -- 1. Verify Python is available
echo [1/5] Checking Python...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.9+ and add to PATH.
    pause
    exit /b 1
)
python --version
echo [OK] Python found.

:: Check Python dependencies from requirements.txt
python -c "import fastapi, ultralytics, cv2, pytesseract" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Python packages missing. Installing from requirements.txt...
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install Python dependencies from requirements.txt.
        pause
        exit /b 1
    )
)
echo [OK] Python dependencies ready.
echo.

:: -- 2. Verify Node.js is available
echo [2/5] Checking Node.js...
node --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js not found in PATH. Please install Node.js 18+ and add to PATH.
    pause
    exit /b 1
)
node --version
echo [OK] Node.js found.
echo.

:: -- 3. Verify node_modules exist
echo [3/5] Checking Node dependencies...
if not exist "node_modules" (
    echo [INFO] node_modules not found. Running npm install...
    call npm install
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
)
echo [OK] Node dependencies ready.
echo.

:: -- 4. Start Python ANPR Inference Server
echo [4/5] Starting Python ANPR Inference Server (port 5001)...
start "ANPR Python Server (port 5001)" cmd /k "python scripts\anpr_server.py"
echo [INFO] Waiting 8 seconds for ANPR server startup...
timeout /t 8 /nobreak >nul
echo.

:: -- 5. Start Node.js Dashboard Server
echo [5/5] Starting Node.js Dashboard Server (port 3000)...
start "Dashboard Node.js Server (port 3000)" cmd /k "node server.js"
echo [INFO] Waiting 3 seconds for Node.js startup...
timeout /t 3 /nobreak >nul

echo.
echo ============================================================
echo   SYSTEM STARTUP COMPLETE
echo ============================================================
echo.
echo   ANPR Inference API:  http://127.0.0.1:5001
echo   Dashboard UI:        https://127.0.0.1:3000
echo   Health Check:        http://127.0.0.1:5001/health
echo   ANPR Config API:     http://127.0.0.1:5001/config
echo   Diagnostics Panel:   https://127.0.0.1:3000/dashboard.html
echo.
echo   NOTE: Your browser may show a security warning for the
echo         self-signed HTTPS certificate. Click "Advanced" and
echo         proceed to continue. This is expected for local dev.
echo.
echo   Press any key to open the dashboard in your browser...
pause >nul
start "" "https://127.0.0.1:3000/dashboard.html"