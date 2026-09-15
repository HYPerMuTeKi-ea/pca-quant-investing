"""Shared defaults for the rolling PCA portfolio project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RETURNS_FILE = PROJECT_ROOT / "data" / "processed" / "returns.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_WINDOW = 252
PCA_K_VALUES = (5, 10, 15, 20)
PRIMARY_PCA_K = 10
ANNUALISATION_FACTOR = 252
ROLLING_VOL_WINDOW = 63
