import logging
import os
import sys
from datetime import datetime

LOG_DIR = "logs"

# Create log file name with timestamp
LOG_PATH = os.path.join(
    LOG_DIR, f"running_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

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
