from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CA_MAX_SCORE = 40.0
EXAM_MAX_SCORE = 60.0
TOTAL_MAX_SCORE = 100.0

DEFAULT_THRESHOLDS = {
    "low": 0.75,
    "medium": 0.50,
}
