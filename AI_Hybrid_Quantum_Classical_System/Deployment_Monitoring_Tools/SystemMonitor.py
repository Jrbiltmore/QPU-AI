# Deployment_Monitoring_Tools/SystemMonitor.py

import logging
from datetime import datetime
import os
import psutil

class SystemMonitor:
    """
    Monitors the system's performance, logs critical system events, and provides alerts for potential issues.
    Designed to help maintain optimal performance and stability of the VR system.
    """

    def __init__(self, log_file='system_monitor.log'):
        """
        Initialize the SystemMonitor with a path to the log file where system events and metrics will be recorded.
        
        Parameters:
            log_file (str): The path to the file where logs should be saved.
        """
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def log_activity(self, message, level='info'):
        """
        Log a specific activity or error at various levels of severity.

        Parameters:
            message (str): The message to log.
            level (str): The severity level of the log ('info', 'warning', 'error', 'critical').
        """
        if level.lower() == 'info':
            logging.info(message)
        elif level.lower() == 'warning':
            logging.warning(message)
        elif level.lower() == 'error':
            logging.error(message)
        elif level.lower() == 'critical':
            logging.critical(message)

    def monitor_resources(self):
        """
        Continuously monitor and log the system's CPU and memory usage.
        """
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        self.log_activity(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%", level='info')

    def alert_high_usage(self):
        """
        Check for high resource usage and alert if thresholds are exceeded.
        """
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        if cpu_usage > 85:
            self.log_activity(f"High CPU usage detected: {cpu_usage}%", level='warning')
        if memory_usage > 85:
            self.log_activity(f"High Memory usage detected: {memory_usage}%", level='warning')

    def system_health_check(self):
        """
        Perform a routine health check of the system, logging the state and alerting on potential issues.
        """
        self.monitor_resources()
        self.alert_high_usage()
        # Additional health checks could be implemented here
        self.log_activity("System health check completed", level='info')
