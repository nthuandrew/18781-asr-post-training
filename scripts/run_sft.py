"""Run config-driven SFT from the repository root."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.sft import main


if __name__ == "__main__":
    raise SystemExit(main())
