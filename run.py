#!/usr/bin/env python3
"""
=============================================================================
  SIH Problem Statement ID: 26127 (BEL)
  City-Wide Indian ANPR Intelligence Platform
  Unified All-in-One Service Launcher: run.py
=============================================================================

Usage:
  python run.py                    # Run full stack (backend + frontend)
  python run.py --no-browser       # Run without opening web browser
  python run.py --separate-windows # Open services in individual terminal windows (Windows)
  python run.py --backend-only     # Run only Python ANPR inference server
  python run.py --frontend-only    # Run only Node.js dashboard server
  python run.py --install-deps     # Check & auto-install missing dependencies first
  python run.py --kill-existing    # Terminate any existing processes on ports 5001 & 3000
=============================================================================
"""

import os
import sys
import time
import socket
import signal
import atexit
import argparse
import threading
import subprocess
import webbrowser
import urllib.request
import urllib.error

# Root project directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default network configuration
ANPR_HOST = "127.0.0.1"
ANPR_PORT = 5001
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 3000

# Ensure UTF-8 output encoding and line buffering on Windows
if sys.platform == "win32":
    os.system("")  # Enable ANSI in Windows terminal
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

# ANSI Color Palette
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"

def print_banner():
    banner = f"""
{CYAN}{BOLD}=============================================================================
  >> SIH Problem Statement ID: 26127 (BEL)
  >> City-Wide Indian ANPR Intelligence Platform
  >> Unified All-in-One System Launcher (run.py)
============================================================================={RESET}
"""
    print(banner)

# List of spawned child processes for cleanup
ACTIVE_PROCESSES = []
CLEANUP_DONE = False

def terminate_process_tree(proc):
    """Cleanly terminate a subprocess and all of its spawned child processes."""
    if proc is None or proc.poll() is not None:
        return
    pid = proc.pid
    try:
        if sys.platform == "win32":
            # /F forces kill, /T kills child tree
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
        else:
            proc.terminate()
            proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

def cleanup():
    """Cleanup handler to ensure no zombie processes or dangling ports remain."""
    global CLEANUP_DONE
    if CLEANUP_DONE:
        return
    CLEANUP_DONE = True
    if ACTIVE_PROCESSES:
        print(f"\n{YELLOW}[SHUTDOWN] Stopping all running services...{RESET}")
        for proc in ACTIVE_PROCESSES:
            terminate_process_tree(proc)
        print(f"{GREEN}[OK] All services stopped cleanly.{RESET}\n")

# Register cleanup handlers
atexit.register(cleanup)

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def is_port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    """Test whether a TCP port is currently open and listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

def kill_process_on_port(port: int):
    """Find and terminate any process listening on the specified port (Windows / Unix)."""
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True, text=True)
            pids = set()
            for line in out.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
                    pids.add(parts[4])
            for pid in pids:
                if pid != "0" and int(pid) != os.getpid():
                    print(f"{YELLOW}[PORT] Terminating existing process (PID {pid}) on port {port}...{RESET}")
                    subprocess.run(["taskkill", "/F", "/PID", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    else:
        try:
            subprocess.run(f"fuser -k {port}/tcp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def check_python_environment(auto_install: bool = False) -> bool:
    """Validate Python runtime and required packages."""
    print(f"{CYAN}[PRE-FLIGHT] Checking Python environment...{RESET}")
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print(f"{RED}[ERROR] Python 3.9+ is required. Found: {sys.version}{RESET}")
        return False
    print(f"  {GREEN}[OK]{RESET} Python {v.major}.{v.minor}.{v.micro} ({sys.executable})")

    # Core requirements check
    missing = []
    for mod in ["fastapi", "uvicorn", "ultralytics", "cv2", "pytesseract"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        print(f"  {YELLOW}[WARN] Missing Python packages: {', '.join(missing)}{RESET}")
        if auto_install:
            print(f"  {CYAN}[INFO] Installing dependencies from requirements.txt...{RESET}")
            req_path = os.path.join(ROOT_DIR, "requirements.txt")
            cmd = [sys.executable, "-m", "pip", "install", "-r", req_path]
            res = subprocess.run(cmd, cwd=ROOT_DIR)
            if res.returncode != 0:
                print(f"{RED}[ERROR] pip install failed with code {res.returncode}.{RESET}")
                return False
            print(f"  {GREEN}[OK]{RESET} Python packages installed successfully.")
        else:
            print(f"  {YELLOW}[TIP] Run with --install-deps to auto-install missing packages.{RESET}")
    else:
        print(f"  {GREEN}[OK]{RESET} Python vision & inference dependencies verified.")

    return True

def check_node_environment(auto_install: bool = False) -> bool:
    """Validate Node.js runtime and node_modules."""
    print(f"{CYAN}[PRE-FLIGHT] Checking Node.js environment...{RESET}")
    try:
        res = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
        node_ver = res.stdout.strip()
        print(f"  {GREEN}[OK]{RESET} Node.js {node_ver} detected.")
    except Exception:
        print(f"{RED}[ERROR] Node.js is not found in PATH.{RESET}")
        print(f"  Please install Node.js 18+ from https://nodejs.org/ and add to PATH.")
        return False

    node_modules = os.path.join(ROOT_DIR, "node_modules")
    if not os.path.exists(node_modules):
        print(f"  {YELLOW}[INFO] node_modules not found.{RESET}")
        if auto_install:
            print(f"  {CYAN}[INFO] Running 'npm install' in {ROOT_DIR}...{RESET}")
            cmd = "npm.cmd" if sys.platform == "win32" else "npm"
            res = subprocess.run([cmd, "install"], cwd=ROOT_DIR)
            if res.returncode != 0:
                print(f"{RED}[ERROR] npm install failed with code {res.returncode}.{RESET}")
                return False
            print(f"  {GREEN}[OK]{RESET} Node modules installed successfully.")
        else:
            print(f"  {YELLOW}[WARN] Please run 'npm install' or re-run with --install-deps.{RESET}")
    else:
        print(f"  {GREEN}[OK]{RESET} node_modules present.")

    return True

def stream_logs(proc, prefix: str, color: str):
    """Continuously read lines from process stdout/stderr and output with a colored prefix."""
    try:
        for line in iter(proc.stdout.readline, ""):
            if line:
                # Strip trailing newline and print
                text = line.rstrip("\r\n")
                if text:
                    print(f"{color}{BOLD}[{prefix}]{RESET} {text}", flush=True)
    except Exception:
        pass

def wait_for_http_service(url: str, timeout: int = 30, service_name: str = "Service") -> bool:
    """Poll an HTTP/HTTPS endpoint until it returns 200 or timeout occurs."""
    start_time = time.time()
    spinner = ["|", "/", "-", "\\"]
    idx = 0
    sys.stdout.write(f"  Waiting for {service_name} to initialize ")
    sys.stdout.flush()

    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ANPR-Launcher/1.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    sys.stdout.write(f"\r  {GREEN}[OK]{RESET} {service_name} is online and ready!         \n")
                    sys.stdout.flush()
                    return True
        except Exception:
            pass

        sys.stdout.write(f"\b{spinner[idx % len(spinner)]}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.5)

    sys.stdout.write(f"\r  {YELLOW}[WARN]{RESET} {service_name} did not respond within {timeout}s (may still be loading). \n")
    sys.stdout.flush()
    return False

def wait_for_port_service(host: str, port: int, timeout: int = 15, service_name: str = "Service") -> bool:
    """Poll a TCP port until a connection succeeds or timeout occurs."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(host, port, timeout=0.5):
            print(f"  {GREEN}[OK]{RESET} {service_name} is listening on {host}:{port}.")
            return True
        time.sleep(0.5)
    print(f"  {YELLOW}[WARN]{RESET} {service_name} port {port} not listening after {timeout}s.")
    return False

def main():
    parser = argparse.ArgumentParser(
        description="SIH-26127 Indian ANPR Multi-Camera Intelligence Platform Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                     # Start both backend and frontend
  python run.py --no-browser        # Launch without automatically opening browser
  python run.py --separate-windows  # Spawn backend & frontend in dedicated consoles (Windows)
  python run.py --kill-existing     # Clear existing port listeners before startup
  python run.py --backend-only      # Run only Python ANPR server
  python run.py --frontend-only     # Run only Node.js server
        """
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the dashboard in browser")
    parser.add_argument("--separate-windows", action="store_true", help="Launch servers in separate terminal windows (Windows only)")
    parser.add_argument("--backend-only", action="store_true", help="Launch only Python ANPR inference microservice (port 5001)")
    parser.add_argument("--frontend-only", action="store_true", help="Launch only Node.js streaming & dashboard server (port 3000)")
    parser.add_argument("--install-deps", action="store_true", help="Auto-install missing Python and Node dependencies")
    parser.add_argument("--skip-checks", action="store_true", help="Skip pre-flight dependency checks")
    parser.add_argument("--kill-existing", action="store_true", help="Automatically kill any processes currently holding ports 5001 or 3000")
    parser.add_argument("--anpr-port", type=int, default=ANPR_PORT, help="Port for Python ANPR server (default: 5001)")
    parser.add_argument("--frontend-port", type=int, default=DASHBOARD_PORT, help="Port for Node.js server (default: 3000)")

    args = parser.parse_args()
    print_banner()

    anpr_script = os.path.join(ROOT_DIR, "scripts", "anpr_server.py")
    node_script = os.path.join(ROOT_DIR, "server.js")

    # 1. Pre-flight Checks
    if not args.skip_checks:
        if not check_python_environment(auto_install=args.install_deps):
            sys.exit(1)
        if not args.backend_only:
            if not check_node_environment(auto_install=args.install_deps):
                sys.exit(1)
        print()

    # 2. Port Conflict & Active Service Handling
    anpr_running = False
    node_running = False

    if not args.frontend_only and is_port_in_use(ANPR_HOST, args.anpr_port):
        if args.kill_existing:
            kill_process_on_port(args.anpr_port)
            time.sleep(0.5)
        else:
            anpr_running = True
            print(f"  {GREEN}[INFO]{RESET} Port {args.anpr_port} is already active. Reusing running ANPR server.")
            print(f"         (Run with --kill-existing to force restart if needed)\n")

    if not args.backend_only and is_port_in_use(DASHBOARD_HOST, args.frontend_port):
        if args.kill_existing:
            kill_process_on_port(args.frontend_port)
            time.sleep(0.5)
        else:
            node_running = True
            print(f"  {GREEN}[INFO]{RESET} Port {args.frontend_port} is already active. Reusing running Node.js server.")
            print(f"         (Run with --kill-existing to force restart if needed)\n")

    # Display configured commands
    print(f"{WHITE}{BOLD}[SERVICE COMMAND LINES]{RESET}")
    if not args.frontend_only:
        anpr_cmd_str = f"{sys.executable} scripts/anpr_server.py"
        status_tag = f"{DIM}(Already Active){RESET}" if anpr_running else f"{DIM}(Port {args.anpr_port}){RESET}"
        print(f"  {CYAN}Backend (ANPR):{RESET}   {anpr_cmd_str}  {status_tag}")
    if not args.backend_only:
        node_cmd_str = f"node server.js"
        status_tag = f"{DIM}(Already Active){RESET}" if node_running else f"{DIM}(Port {args.frontend_port}){RESET}"
        print(f"  {GREEN}Frontend (Node):{RESET}  {node_cmd_str}  {status_tag}")
    print()

    # 3. Launch Modes
    # Option A: Separate Windows (Windows Only)
    if args.separate_windows and sys.platform == "win32":
        print(f"{CYAN}[LAUNCH] Launching services in dedicated terminal windows...{RESET}\n")
        if not args.frontend_only and not anpr_running:
            p1 = subprocess.Popen(
                f'start "ANPR Python Server (Port {args.anpr_port})" cmd /k "{sys.executable} scripts\\anpr_server.py"',
                shell=True, cwd=ROOT_DIR
            )
            time.sleep(2)

        if not args.backend_only and not node_running:
            p2 = subprocess.Popen(
                f'start "Dashboard Node.js Server (Port {args.frontend_port})" cmd /k "node server.js"',
                shell=True, cwd=ROOT_DIR
            )
            time.sleep(2)

        # Print info and optionally open browser
        dashboard_url = f"https://127.0.0.1:{args.frontend_port}/dashboard.html"
        print(f"\n{GREEN}{BOLD}Services dispatched to individual consoles.{RESET}")
        print(f"Dashboard URL: {dashboard_url}\n")
        if not args.no_browser and not args.backend_only:
            webbrowser.open(dashboard_url)
        return

    # Option B: Unified Console with Live Log Streaming
    print(f"{CYAN}[LAUNCH] Starting services with unified output streaming...{RESET}\n")

    anpr_proc = None
    node_proc = None

    # Start Backend (ANPR Server)
    if not args.frontend_only:
        if anpr_running:
            print(f"{CYAN}[1/2] ANPR Server already running on port {args.anpr_port}. Skipping launch.{RESET}\n")
        else:
            print(f"{CYAN}[1/2] Starting Python ANPR Inference Server (Port {args.anpr_port})...{RESET}")
            anpr_cmd = [sys.executable, anpr_script]
            anpr_proc = subprocess.Popen(
                anpr_cmd,
                cwd=ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )
            ACTIVE_PROCESSES.append(anpr_proc)

            # Start background log thread
            t1 = threading.Thread(target=stream_logs, args=(anpr_proc, "ANPR-BACKEND", CYAN), daemon=True)
            t1.start()

            # Wait for ANPR server health endpoint
            health_url = f"http://{ANPR_HOST}:{args.anpr_port}/health"
            wait_for_http_service(health_url, timeout=25, service_name="ANPR Inference Microservice")
            print()

    # Start Frontend (Node.js Dashboard Server)
    if not args.backend_only:
        if node_running:
            print(f"{GREEN}[2/2] Node.js Dashboard Server already running on port {args.frontend_port}. Skipping launch.{RESET}\n")
        else:
            print(f"{GREEN}[2/2] Starting Node.js Web Dashboard Server (Port {args.frontend_port})...{RESET}")
            node_cmd = ["node", node_script]
            node_proc = subprocess.Popen(
                node_cmd,
                cwd=ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )
            ACTIVE_PROCESSES.append(node_proc)

            # Start background log thread
            t2 = threading.Thread(target=stream_logs, args=(node_proc, "NODE-FRONTEND", GREEN), daemon=True)
            t2.start()

            # Wait for Node port
            wait_for_port_service(DASHBOARD_HOST, args.frontend_port, timeout=10, service_name="Node.js Dashboard")
            print()

    # 4. System Summary Banner
    summary = f"""
{GREEN}{BOLD}=============================================================================
  >> SYSTEM STARTUP COMPLETE & READY
============================================================================={RESET}

  {BOLD}* Web Dashboard UI:{RESET}        {WHITE}https://127.0.0.1:{args.frontend_port}/dashboard.html{RESET}
  {BOLD}* Camera Node Streamer:{RESET}    {WHITE}https://127.0.0.1:{args.frontend_port}/camera.html{RESET}
  {BOLD}* ANPR Inference API:{RESET}      {WHITE}http://127.0.0.1:{args.anpr_port}/detect{RESET}
  {BOLD}* ANPR Health Check:{RESET}       {WHITE}http://127.0.0.1:{args.anpr_port}/health{RESET}
  {BOLD}* Pipeline Config API:{RESET}     {WHITE}http://127.0.0.1:{args.anpr_port}/config{RESET}

  {YELLOW}{BOLD}NOTE:{RESET} {YELLOW}Your browser may display a self-signed HTTPS certificate warning.
        Click "Advanced" -> "Proceed" to continue. HTTPS is required
        by browsers for WebRTC camera access.{RESET}

  {DIM}Press Ctrl+C at any time to gracefully stop all services.{RESET}
{GREEN}{BOLD}============================================================================={RESET}
"""
    print(summary)

    # 5. Open Web Browser
    if not args.no_browser and not args.backend_only:
        dashboard_url = f"https://127.0.0.1:{args.frontend_port}/dashboard.html"
        print(f"{CYAN}[BROWSER] Opening dashboard in default web browser...{RESET}\n")
        webbrowser.open(dashboard_url)

    # 6. Keep main thread alive and monitor child processes
    try:
        while True:
            # Check if any process died unexpectedly
            if anpr_proc and anpr_proc.poll() is not None:
                code = anpr_proc.returncode
                print(f"\n{RED}[ERROR] ANPR server process terminated unexpectedly with exit code {code}.{RESET}")
                break
            if node_proc and node_proc.poll() is not None:
                code = node_proc.returncode
                print(f"\n{RED}[ERROR] Node.js server process terminated unexpectedly with exit code {code}.{RESET}")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[INTERRUPT] Received Ctrl+C keyboard interrupt.{RESET}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
