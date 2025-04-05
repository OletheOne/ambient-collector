import os
import numpy as np
import time
from datetime import datetime
import simpleaudio as sa
import random

class AmbientSonifier:
    def __init__(self):
        self.data_path = "ambient_data"
        self.output_path = "ambient_audio"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
    def read_ambient_files(self):
        # Same as in the visualizer
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
    
    def generate_tone(self, frequency, duration, volume=0.5):
        # Generate a sine wave tone
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(frequency * t * 2 * np.pi)
        
        # Apply fade in/out
        fade_duration = int(sample_rate * 0.1)
        fade_in = np.linspace(0, 1, fade_duration)
        fade_out = np.linspace(1, 0, fade_duration)
        
        tone[:fade_duration] *= fade_in
        tone[-fade_duration:] *= fade_out
        
        # Adjust volume
        tone = tone * volume
        
        # Convert to 16-bit data
        audio = tone * (2**15 - 1) / np.max(np.abs(tone))
        audio = audio.astype(np.int16)
        
        return audio
    
    def sonify_data(self, data_points):
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
        
        # Base frequency on average CPU usage (higher CPU = higher pitch)
        base_freq = 220 + (np.mean(cpu_values) * 4)  # A3 + scaling
        
        # Create a sequence of notes based on data
        total_duration = 0
        combined_audio = np.array([], dtype=np.int16)
        
        for i, (cpu, mem) in enumerate(zip(cpu_values, mem_values)):
            # Vary frequency based on CPU
            freq_modifier = 1 + (cpu - 50) / 100  # Normalize around 1.0
            frequency = base_freq * max(0.5, min(2.0, freq_modifier))
            
            # Duration based on memory usage
            duration = 0.2 + (mem / 200)  # 0.2 to 0.7 seconds
            
            # Volume based on position in sequence
            volume = 0.3 + (i / len(cpu_values)) * 0.5
            
            # Generate tone
            tone = self.generate_tone(frequency, duration, volume)
            combined_audio = np.append(combined_audio, tone)
            
            # Add a small silence between notes
            silence_duration = 0.05
            silence = np.zeros(int(44100 * silence_duration), dtype=np.int16)
            combined_audio = np.append(combined_audio, silence)
            
            total_duration += duration + silence_duration
        
        # Add some harmonic tones based on overall system activity
        if len(cpu_values) > 2:
            harmony_freq = base_freq * 1.5  # Perfect fifth
            harmony_duration = total_duration / 2
            harmony_volume = 0.15
            harmony = self.generate_tone(harmony_freq, harmony_duration, harmony_volume)
            
            # Mix with main audio (at the beginning)
            harmony_padded = np.pad(harmony, (0, max(0, len(combined_audio) - len(harmony))), 'constant')
            combined_audio = combined_audio + harmony_padded[:len(combined_audio)]
        
        # Save as WAV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_path, f"ambient_sound_{timestamp}.wav")
        
       # Play the sound
        try:
            # Make sure audio data is properly formatted
            # simpleaudio expects: data, num_channels, bytes_per_sample, sample_rate
            play_obj = sa.play_buffer(combined_audio, 1, 2, 44100)
            print(f"Playing ambient sound based on system state...")
            play_obj.wait_done()
        except Exception as e:
            print(f"Error playing audio: {e}")
        
        # Save as WAV file using scipy
        try:
            from scipy.io import wavfile
            wavfile.write(output_file, 44100, combined_audio)
            print(f"Saved audio to {output_file}")
        except ImportError:
            print(f"Note: To save as WAV, you need to install scipy with 'pip install scipy'")
        except Exception as e:
            print(f"Error saving audio: {e}")
        
        return combined_audio

if __name__ == "__main__":
    sonifier = AmbientSonifier()
    data_points = sonifier.read_ambient_files()
    sonifier.sonify_data(data_points)