import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

logger = logging.getLogger("mlProjectLogger")
logger.setLevel(logging.INFO)
logger.propagate = True

if not logger.handlers:
    # delay=True prevents the handler from creating the file until first write
    file_handler = logging.FileHandler(LOG_PATH, delay=True)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
