import logging
import os
import sys

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "debug_logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format=logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

general_logger = logging.getLogger("media_bias_detectionLogger")
