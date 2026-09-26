"""Smoke-run every script in examples/ and fail if any of them errors.

Examples that download data are skipped (not failed) when the network is
unavailable, and examples with optional dependencies are skipped when the
dependency is not installed.
"""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 300

NEEDS_NETWORK = {"demo_real_world.py", "climate_forecasting.py", "finance_stock_volatility.py"}
OPTIONAL_DEPS = {"finance_stock_volatility.py": "yfinance"}
NETWORK_ERRORS = ("URLError", "ConnectionError", "HTTPError", "Temporary failure in name resolution")


def main():
    env = dict(os.environ, MPLBACKEND="Agg")
    failed, skipped = [], []

    for path in sorted((ROOT / "examples").rglob("*.py")):
        name = path.relative_to(ROOT)
        dep = OPTIONAL_DEPS.get(path.name)
        if dep and importlib.util.find_spec(dep) is None:
            skipped.append(f"{name} (needs {dep})")
            continue

        start = time.time()
        try:
            proc = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env,
                                  capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            failed.append(f"{name} (timed out after {TIMEOUT}s)")
            continue

        elapsed = time.time() - start
        if proc.returncode == 0:
            print(f"ok      {elapsed:5.1f}s  {name}")
        elif path.name in NEEDS_NETWORK and any(e in proc.stderr for e in NETWORK_ERRORS):
            skipped.append(f"{name} (network unavailable)")
        else:
            failed.append(str(name))
            print(f"FAILED  {elapsed:5.1f}s  {name}")
            print("\n".join(proc.stderr.splitlines()[-15:]))

    for item in skipped:
        print(f"skipped         {item}")
    if failed:
        print(f"\n{len(failed)} example(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
