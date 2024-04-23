
# /Deployment_Monitoring_Tools/system_monitoring_tool.py

import psutil

def system_stats():
    # Get CPU usage
    cpu_usage = psutil.cpu_percent(interval=1)
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    # Get disk usage
    disk_usage = psutil.disk_usage('/').percent
    return {
        'CPU Usage (%)': cpu_usage,
        'Memory Usage (%)': memory_usage,
        'Disk Usage (%)': disk_usage
    }

# Example usage
if __name__ == '__main__':
    stats = system_stats()
    print("System Monitoring Stats:", stats)
