import logging
import os
import sys

LOG_DIR = "logs"

LOG_PATH = os.path.join(LOG_DIR, "running_log.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),  # So we can see the log in the terminal
    ],
)

logger = logging.getLogger("mlProjectLogger")
