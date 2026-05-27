"""Đường dẫn gốc dự án (charts/, data/ nằm ở repo root)."""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
CHARTS_DIR = PROJECT_ROOT / "charts"
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHART_PATH = CHARTS_DIR / "chart_output.png"
