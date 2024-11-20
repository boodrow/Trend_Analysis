# logger.py

import logging

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture detailed logs

# Clear any existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Create console handler and set level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter and add it to the console handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)