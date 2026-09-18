#!/usr/bin/env python3
import subprocess, sys, venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"

def run(*args): subprocess.check_call(list(args))

run("git", "submodule", "update", "--init", "--recursive")

if not VENV.exists():
    venv.create(VENV, with_pip=True)

# Cross-platform: pick the right python path inside the venv
py = VENV / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")
py = str(py)

run(py, "-m", "pip", "install", "--upgrade", "pip")
run(py, "-m", "pip", "install",
    "-e", str(ROOT),
    "-e", str(ROOT / "libs/robotarium"),
    "--config-settings", "editable_mode=compat")

print("Done.")