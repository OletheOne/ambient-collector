import os
import numpy as np
import time
from datetime import datetime
import simpleaudio as sa

class GentleAmbientSonifier:
    def __init__(self):
        self.data_path = "ambient_data"
        
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
    
    def generate_gentle_tone(self, frequency, duration, volume=0.3):
        # Generate a softer sine wave tone
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Use multiple harmonics with decreasing amplitude for a softer sound
        tone = np.sin(frequency * t * 2 * np.pi) * 0.6
        tone += np.sin(frequency * 2 * t * 2 * np.pi) * 0.2
        tone += np.sin(frequency * 3 * t * 2 * np.pi) * 0.1
        
        # Apply longer fade in/out
        fade_duration = int(sample_rate * 0.3)  # 300ms fade
        if len(tone) > 2 * fade_duration:
            fade_in = np.linspace(0, 1, fade_duration)
            fade_out = np.linspace(1, 0, fade_duration)
            
            tone[:fade_duration] *= fade_in
            tone[-fade_duration:] *= fade_out
        
        # Adjust volume (lower than before)
        tone = tone * volume
        
        # Convert to 16-bit data
        audio = tone * (2**15 - 1) / np.max(np.abs(tone))
        audio = audio.astype(np.int16)
        
        return audio
    
    def sonify_data_gently(self, data_points):
        if not data_points:
            print("No data points found")
            return
        
        # Extract metrics
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
        
        # Use a calming pentatonic scale (A, C, D, E, G)
        pentatonic = [220.00, 261.63, 293.66, 329.63, 392.00]
        
        # Create a gentle sequence of notes based on data
        combined_audio = np.array([], dtype=np.int16)
        
        for i, (cpu, mem) in enumerate(zip(cpu_values, mem_values)):
            # Choose note from pentatonic scale based on CPU
            note_index = int((cpu / 100) * (len(pentatonic) - 1))
            note_index = max(0, min(len(pentatonic) - 1, note_index))
            frequency = pentatonic[note_index]
            
            # Longer duration for more gentle feel
            duration = 0.7 + (mem / 200)  # 0.7 to 1.2 seconds
            
            # Lower volume for gentleness
            volume = 0.15 + (i / len(cpu_values)) * 0.15  # 0.15 to 0.3
            
            # Generate gentle tone
            tone = self.generate_gentle_tone(frequency, duration, volume)
            combined_audio = np.append(combined_audio, tone)
            
            # Add a longer silence between notes
            silence_duration = 0.2
            silence = np.zeros(int(44100 * silence_duration), dtype=np.int16)
            combined_audio = np.append(combined_audio, silence)
        
        # Play the sound
        play_obj = sa.play_buffer(combined_audio, 1, 2, 44100)
        print(f"Playing gentle ambient sound based on system state...")
        play_obj.wait_done()
        
        return combined_audio

if __name__ == "__main__":
    sonifier = GentleAmbientSonifier()
    data_points = sonifier.read_ambient_files()
    sonifier.sonify_data_gently(data_points)