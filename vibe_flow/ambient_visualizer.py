import os
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime

class AmbientVisualizer:
    def __init__(self):
        self.data_path = "ambient_data"
        self.output_path = "ambient_visuals"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    
    def read_ambient_files(self):
        data_points = []
        for filename in os.listdir(self.data_path):
            if filename.startswith("ambient_") and filename.endswith(".txt"):
                file_path = os.path.join(self.data_path, filename)
                data_point = {}
                with open(file_path, "r") as f:
                    for line in f:
                        if ":" in line:
                            key, value = line.strip().split(":", 1)
                            data_point[key.strip()] = value.strip()
                data_points.append(data_point)
        return data_points
    
    def generate_abstract_visual(self, data_points):
        if not data_points:
            print("No data points found")
            return
        
        # Create a figure with a black background
        plt.figure(figsize=(10, 8), facecolor='black')
        ax = plt.subplot(111, facecolor='black')
        
        # Extract CPU and memory usage from data points
        cpu_values = []
        mem_values = []
        for dp in data_points:
            try:
                cpu = float(dp.get("cpu_percent", 0))
                mem = float(dp.get("memory_percent", 0))
                cpu_values.append(cpu)
                mem_values.append(mem)
            except (ValueError, TypeError):
                continue
        
        # Generate abstract art based on system metrics
        for i in range(len(cpu_values)):
            # Color based on CPU/memory ratio
            color_intensity = cpu_values[i] / 100.0
            color = plt.cm.viridis(color_intensity)
            
            # Size based on memory usage
            size = mem_values[i] * 5
            
            # Position with some randomness
            x = i + random.uniform(-0.3, 0.3)
            y = cpu_values[i] + random.uniform(-5, 5)
            
            # Draw element
            ax.scatter(x, y, s=size, color=color, alpha=0.7)
            
            # Add connecting lines with gradient
            if i > 0:
                plt.plot([i-1, i], [cpu_values[i-1], cpu_values[i]], 
                         color=color, alpha=0.3, linewidth=1.5)
        
        # Add ambient elements
        for _ in range(50):
            x = random.uniform(0, len(cpu_values) - 1)
            y = random.uniform(0, 100)
            size = random.uniform(5, 20)
            color = plt.cm.magma(random.random())
            ax.scatter(x, y, s=size, color=color, alpha=0.2)
        
        # Remove axes for aesthetic appeal
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_path, f"ambient_visual_{timestamp}.png")
        plt.savefig(output_file, facecolor='black', bbox_inches='tight')
        plt.close()
        
        print(f"Created abstract visualization at {output_file}")

if __name__ == "__main__":
    visualizer = AmbientVisualizer()
    data_points = visualizer.read_ambient_files()
    visualizer.generate_abstract_visual(data_points)