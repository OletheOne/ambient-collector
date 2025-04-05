import time
import os
from datetime import datetime
import platform
import psutil

class AmbientCollector:
    def __init__(self):
        self.data_path = "ambient_data"
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
    def collect_system_ambience(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Collect basic system information
        system_info = {
            "timestamp": timestamp,
            "platform": platform.platform(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "battery": psutil.sensors_battery().percent if hasattr(psutil.sensors_battery(), "percent") else None,
            "running_processes": len(list(psutil.process_iter()))
        }
        
        # Save to file
        filename = os.path.join(self.data_path, f"ambient_{timestamp}.txt")
        with open(filename, "w") as f:
            for key, value in system_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Ambient data collected and saved to {filename}")
        return system_info

# Add this to the bottom of your ambient_collector.py file, replacing the existing if __name__ block:

if __name__ == "__main__":
    collector = AmbientCollector()
    try:
        print("Collecting ambient data every 5 seconds. Press Ctrl+C to stop.")
        for _ in range(10):  # Collect 10 data points
            collector.collect_system_ambience()
            time.sleep(5)
    except KeyboardInterrupt:
        print("Data collection stopped.")