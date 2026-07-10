# Delegates to the canonical root-level entry point.
# Run from project root: python run_ingestion.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_ingestion import main  # noqa: E402

if __name__ == "__main__":
    main()
