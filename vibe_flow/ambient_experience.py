import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import simpleaudio as sa
import time
from datetime import datetime
import threading

class AmbientExperience:
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
        # Same as in gentle_ambient_sonifier.py
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        tone = np.sin(frequency * t * 2 * np.pi) * 0.6
        tone += np.sin(frequency * 2 * t * 2 * np.pi) * 0.2
        tone += np.sin(frequency * 3 * t * 2 * np.pi) * 0.1
        
        fade_duration = int(sample_rate * 0.3)
        if len(tone) > 2 * fade_duration:
            fade_in = np.linspace(0, 1, fade_duration)
            fade_out = np.linspace(1, 0, fade_duration)
            
            tone[:fade_duration] *= fade_in
            tone[-fade_duration:] *= fade_out
        
        tone = tone * volume
        
        audio = tone * (2**15 - 1) / np.max(np.abs(tone))
        audio = audio.astype(np.int16)
        
        return audio
    
    def create_integrated_experience(self, data_points):
        if not data_points:
            print("No data points found")
            return
        
        # Extract metrics
        cpu_values = []
        mem_values = []
        processes = []
        
        for dp in data_points:
            try:
                cpu = float(dp.get("cpu_percent", 0))
                mem = float(dp.get("memory_percent", 0))
                proc = int(dp.get("running_processes", 0))
                
                cpu_values.append(cpu)
                mem_values.append(mem)
                processes.append(proc)
            except (ValueError, TypeError):
                continue
        
        # Prepare audio sequence
        pentatonic = [220.00, 261.63, 293.66, 329.63, 392.00]
        combined_audio = np.array([], dtype=np.int16)
        note_times = []  # To track when notes occur for visual sync
        current_time = 0
        
        for i, (cpu, mem) in enumerate(zip(cpu_values, mem_values)):
            note_index = int((cpu / 100) * (len(pentatonic) - 1))
            note_index = max(0, min(len(pentatonic) - 1, note_index))
            frequency = pentatonic[note_index]
            
            duration = 0.7 + (mem / 200)
            volume = 0.15 + (i / len(cpu_values)) * 0.15
            
            tone = self.generate_gentle_tone(frequency, duration, volume)
            combined_audio = np.append(combined_audio, tone)
            
            # Record the time this note occurs
            note_times.append((current_time, duration, frequency, cpu, mem))
            current_time += duration
            
            silence_duration = 0.2
            silence = np.zeros(int(44100 * silence_duration), dtype=np.int16)
            combined_audio = np.append(combined_audio, silence)
            current_time += silence_duration
        
        # Start audio in a separate thread
        def play_audio():
            play_obj = sa.play_buffer(combined_audio, 1, 2, 44100)
            play_obj.wait_done()
        
        audio_thread = threading.Thread(target=play_audio)
        audio_thread.daemon = True
        
        # Set up the visualization
        fig, ax = plt.figure(figsize=(10, 8), facecolor='black'), plt.subplot(111, facecolor='black')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        # Custom colormap for a gentle, calming feel
        colors = [(0.05, 0.05, 0.1), (0.1, 0.2, 0.3), (0.2, 0.3, 0.5), (0.3, 0.5, 0.7), (0.5, 0.7, 0.9)]
        cmap = LinearSegmentedColormap.from_list("calm_ocean", colors, N=100)
        
        # Create initial gradient background
        gradient = np.linspace(0, 1, 100).reshape(10, 10)
        img = ax.imshow(gradient, cmap=cmap, aspect='auto', extent=[-1, 1, -1, 1], alpha=0.8)
        
        # No axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        
        # Particle system for visualization
        n_particles = 50
        particles = ax.scatter([], [], s=[], c=[], alpha=0.7, cmap=cmap)
        
        def init():
            particles.set_offsets(np.empty((0, 2)))
            particles.set_sizes(np.array([]))
            particles.set_array(np.array([]))
            return particles,
        
        # Animation update function
        particle_positions = np.random.uniform(-1, 1, (n_particles, 2))
        particle_velocities = np.random.uniform(-0.01, 0.01, (n_particles, 2))
        particle_sizes = np.random.uniform(20, 100, n_particles)
        particle_colors = np.random.uniform(0, 1, n_particles)
        active_notes = []
        
        def update(frame):
            nonlocal particle_positions, particle_velocities, particle_sizes, particle_colors, active_notes
            
            # Update current time
            current_time = frame / 30  # Assuming 30fps
            
            # Check for notes that should be active
            for note_start, duration, freq, cpu, mem in note_times:
                if note_start <= current_time <= note_start + duration:
                    if (note_start, duration, freq) not in active_notes:
                        active_notes.append((note_start, duration, freq))
                        # Boost some particles when a note starts
                        boost_indices = np.random.choice(n_particles, 5)
                        particle_velocities[boost_indices] += np.random.uniform(-0.05, 0.05, (5, 2))
                        particle_colors[boost_indices] = cpu / 100  # Color based on CPU
                        particle_sizes[boost_indices] = 100 + mem  # Size based on memory
                elif note_start + duration < current_time and (note_start, duration, freq) in active_notes:
                    active_notes.remove((note_start, duration, freq))
            
            # Update particle positions with gentle drift
            particle_positions += particle_velocities
            
            # Contain particles within boundaries with soft bounce
            out_of_bounds = np.abs(particle_positions) > 1
            particle_velocities[out_of_bounds] *= -0.7
            
            # Add slight attraction to center to keep particles from drifting too far
            center_vector = -particle_positions * 0.001
            particle_velocities += center_vector
            
            # Small random movement
            particle_velocities += np.random.uniform(-0.001, 0.001, (n_particles, 2))
            
            # Limit velocity
            velocity_magnitude = np.sqrt(np.sum(particle_velocities**2, axis=1))
            too_fast = velocity_magnitude > 0.02
            if np.any(too_fast):
                particle_velocities[too_fast] *= 0.02 / velocity_magnitude[too_fast].reshape(-1, 1)
            
            # Slowly change colors and sizes for ambient effect
            particle_colors += np.random.uniform(-0.01, 0.01, n_particles)
            particle_colors = np.clip(particle_colors, 0, 1)
            
            particle_sizes += np.random.uniform(-1, 1, n_particles)
            particle_sizes = np.clip(particle_sizes, 20, 150)
            
            # Boost sizes for active notes
            if active_notes:
                particle_sizes *= 1.01
            
            # Update particles
            particles.set_offsets(particle_positions)
            particles.set_sizes(particle_sizes)
            particles.set_array(particle_colors)
            
            # Update background gradient
            if frame % 5 == 0:  # Every 5 frames to save computation
                gradient = np.random.normal(0.5, 0.1, (10, 10))
                for i, (_, _, freq, _, _) in enumerate(active_notes):
                    # Add wave patterns based on active frequencies
                    x = np.linspace(-np.pi, np.pi, 10)
                    y = np.linspace(-np.pi, np.pi, 10)
                    X, Y = np.meshgrid(x, y)
                    wave = 0.1 * np.sin(X * (freq/100) + frame/10) * np.cos(Y * (freq/150) + frame/12)
                    gradient += wave
                
                gradient = np.clip(gradient, 0, 1)
                img.set_array(gradient)
            
            return particles, img
        
        # Calculate animation duration based on audio
        anim_duration = current_time  # seconds
        frames = int(anim_duration * 30)  # 30fps
        
        ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, 
                                      interval=33.3, blit=True)
        
        # Start audio and show visualization
        print("Starting ambient experience...")
        audio_thread.start()
        plt.show()
        
        # Wait for audio to finish
        audio_thread.join()
        print("Ambient experience complete.")

if __name__ == "__main__":
    experience = AmbientExperience()
    data_points = experience.read_ambient_files()
    experience.create_integrated_experience(data_points)