import os
from pathlib import Path

import yaml

# --- Constants ---
REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DATA_DIR = Path(os.environ.get("FACESORTER_DATA_DIR",
                                   Path.home() / ".facesorter"))
SCAN_CACHE_DB = APP_DATA_DIR / "scan_cache.db"
CROP_DIR = APP_DATA_DIR / "crops"
TEMP_UPLOAD_DIR = APP_DATA_DIR / "uploads"
OUTPUT_DIR = str(REPO_ROOT / "sorted_output")


class Config:
    """Loads default settings from config.yaml at the repo root."""

    def __init__(self, config_file=None):
        config_file = config_file or REPO_ROOT / "config.yaml"
        try:
            with open(config_file, "r") as f:
                self.settings = yaml.safe_load(f) or {}
        except FileNotFoundError:
            self.settings = {}

    def get(self, key, default=None):
        """Retrieves a value by dot-notation key, e.g. 'clustering.eps'."""
        value = self.settings
        try:
            for k in key.split("."):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


config = Config()
