from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# CONFIG_FILE_PATH = Path("config/config.yaml")
# relative to where the script is executed

CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"
# This version is independent of the working directory
PARAMS_FILE_PATH = PROJECT_ROOT / "config" / "params.yaml"
SCHEMA_FILE_PATH = PROJECT_ROOT / "config" / "schema.yaml"
