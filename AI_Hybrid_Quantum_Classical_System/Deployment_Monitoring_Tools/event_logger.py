
# /Deployment_Monitoring_Tools/event_logger.py

import logging
from datetime import datetime

# Setup logging configuration
logging.basicConfig(filename='system_events.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EventLogger:
    def __init__(self):
        self.logger = logging.getLogger('EventLogger')

    def log_event(self, message):
        # Log an event with the current timestamp
        self.logger.info(message)

# Example usage
if __name__ == '__main__':
    logger = EventLogger()
    logger.log_event('System startup at ' + str(datetime.now()))
    logger.log_event('System shutdown at ' + str(datetime.now()))
