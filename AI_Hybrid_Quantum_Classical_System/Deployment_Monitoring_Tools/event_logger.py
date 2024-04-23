import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import os

# Function to set up logging with rotation
def setup_logger():
    log_filename = 'system_events.log'
    logger = logging.getLogger('EventLogger')
    logger.setLevel(logging.INFO)  # Default to INFO level

    # Check if a debug mode is set in the environment variables
    if os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 't'):
        logger.setLevel(logging.DEBUG)

    # Log rotation setup
    handler = TimedRotatingFileHandler(log_filename, when="midnight", interval=1, backupCount=7)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(handler)
    return logger

class EventLogger:
    def __init__(self):
        self.logger = setup_logger()

    def log_event(self, message):
        # Log an event with the current timestamp
        self.logger.info(message)

    def log_debug(self, message):
        # Log a debug message, if the level allows
        self.logger.debug(message)

# Example usage
if __name__ == '__main__':
    logger = EventLogger()
    logger.log_event('System startup at ' + str(datetime.now()))
    logger.log_debug('Debugging startup sequence')
    logger.log_event('System shutdown at ' + str(datetime.now()))
