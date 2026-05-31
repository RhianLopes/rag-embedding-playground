import glob, subprocess, sys, os

notebooks = sorted(glob.glob("**/*.ipynb", recursive=True))
notebooks = [n for n in notebooks if ".venv" not in n and ".ipynb_checkpoints" not in n and "test_render" not in n]

python = os.path.join(".venv", "Scripts", "python.exe")

for nb in notebooks:
    print(f"Converting: {nb} ...", flush=True)
    result = subprocess.run(
        [python, "-m", "nbconvert", "--to", "html",
         "--execute", "--allow-errors",
         "--ExecutePreprocessor.timeout=180",
         nb],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  OK")
    else:
        # nbconvert may still write HTML even on non-zero exit
        stderr_tail = result.stderr.strip().splitlines()
        last = stderr_tail[-1] if stderr_tail else ""
        if "Writing" in last:
            print(f"  OK (with errors in cells)")
        else:
            print(f"  FAILED: {last}")
