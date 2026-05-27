"""Điểm vào tương thích: streamlit run app.py"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import apps.streamlit_app  # noqa: F401
