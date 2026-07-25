import sys
from pathlib import Path

# make the src/ modules importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
