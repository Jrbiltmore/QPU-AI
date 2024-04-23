import psutil
import logging
from logging.handlers import TimedRotatingFileHandler
import time

def setup_logger():
    """Set up logging configuration."""
    logger = logging.getLogger('SystemMonitor')
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler('system_monitoring.log', when='midnight', backupCount=7)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def system_stats():
    """Gather system statistics."""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    disk_usage = psutil.disk_usage('/').percent
    return {
        'CPU Usage (%)': cpu_usage,
        'Memory Usage (%)': memory_usage,
        'Disk Usage (%)': disk_usage
    }

def monitor_system(interval=60):
    """Monitor system at specified intervals and log stats."""
    logger = setup_logger()
    logger.info("Starting system monitoring.")
    try:
        while True:
            stats = system_stats()
            logger.info(f"System Stats: {stats}")
            if stats['CPU Usage (%)'] > 90:
                logger.warning("High CPU usage detected!")
            if stats['Memory Usage (%)'] > 90:
                logger.warning("High memory usage detected!")
            if stats['Disk Usage (%)'] > 90:
                logger.warning("High disk usage detected!")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Stopping system monitoring.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

# Example usage
if __name__ == '__main__':
    monitor_system(300)  # Monitor every 5 minutes
