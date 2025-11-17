from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# CONFIG_FILE_PATH = Path("config/config.yaml")
# relative to where the script is executed

CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"
# This version uses absolute path based on project root
PARAMS_FILE_PATH = PROJECT_ROOT / "config" / "params.yaml"
SCHEMA_FILE_PATH = PROJECT_ROOT / "config" / "schema.yaml"
