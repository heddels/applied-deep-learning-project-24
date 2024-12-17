"""Logging configuration for the entire project.

Sets up a centralized logger that:
1. Writes to both file and console
2. Includes timestamps and log levels
3. Creates log directory if needed

Format: [Time: Level: Module: Message]
Location: debug_logs/running_logs.log
Default Level: WARNING
"""

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
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
)

general_logger = logging.getLogger("media_bias_detectionLogger")
