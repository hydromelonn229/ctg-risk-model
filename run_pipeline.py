"""Rebuild the processed tables and figures from the downloaded dataset."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
STEPS = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_preprocessing.py",
    "notebooks/03_features.py",
    "notebooks/02_modelling.py",
    "notebooks/03_explainability.py",
]

for relative_path in STEPS:
    script = ROOT / relative_path
    print(f"\n--- Running {relative_path} ---")
    if script.suffix == ".ipynb":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                str(script),
            ],
            cwd=ROOT,
            check=True,
        )
    else:
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

print("\nPipeline completed successfully.")
