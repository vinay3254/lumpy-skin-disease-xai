from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_CSV = DATA_DIR / "splits.csv"
SPLITS_C3_CSV = DATA_DIR / "splits_c3.csv"
OUT_DIR = ROOT / "outputs"
RUNS_DIR = OUT_DIR / "runs"

IMG_SIZE = 224
SEED = 42
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASS_NAMES = {2: ["healthy", "lsd"], 3: ["healthy", "lsd", "other"]}
