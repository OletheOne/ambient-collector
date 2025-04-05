import time
import os
import psutil
import platform
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import simpleaudio as sa
import threading
from matplotlib.colors import LinearSegmentedColormap

class DigitalMoodRing:
    def __init__(self):
        self.data_path = "mood_data"
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        # Data storage
        self.current_session = []
        
        # Create a color palette based on emotional states
        self.mood_colors = {
            'calm': [(0.0, 0.1, 0.2), (0.1, 0.3, 0.5), (0.2, 0.5, 0.8)],  # Blues
            'busy': [(0.2, 0.0, 0.0), (0.5, 0.1, 0.1), (0.8, 0.2, 0.2)],  # Reds
            'balanced': [(0.0, 0.2, 0.0), (0.1, 0.5, 0.1), (0.2, 0.8, 0.3)],  # Greens
            'distracted': [(0.2, 0.1, 0.0), (0.5, 0.3, 0.0), (0.8, 0.5, 0.0)]  # Oranges
        }
        
        # Sound parameters for different moods
        self.mood_sounds = {
            'calm': {'scale': [220.00, 277.18, 329.63, 369.99, 440.00],  # A minor pentatonic
                    'tempo': 0.8, 'volume': 0.2},
            'busy': {'scale': [261.63, 311.13, 349.23, 392.00, 466.16],  # C major pentatonic
                    'tempo': 0.4, 'volume': 0.3},
            'balanced': {'scale': [246.94, 293.66, 349.23, 392.00, 493.88],  # B minor pentatonic
                        'tempo': 0.6, 'volume': 0.25},
            'distracted': {'scale': [233.08, 277.18, 311.13, 369.99, 415.30],  # G# minor pentatonic
                        'tempo': 0.5, 'volume': 0.28}
        }
    
    def collect_system_state(self):
        """Collect current system state metrics"""
        timestamp = datetime.now()
        
        # Get CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory metrics
        memory = psutil.virtual_memory()
        mem_percent = memory.percent
        
        # Get process information
        num_processes = len(list(psutil.process_iter()))
        
        # Build the state object
        state = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'cpu_percent': cpu_percent,
            'memory_percent': mem_percent,
            'num_processes': num_processes
        }
        
        self.current_session.append(state)
        return state
    
    def determine_mood(self):
        """Analyze system state to determine the current 'mood'"""
        # Use the most recent 5 states or all if fewer
        recent_states = self.current_session[-5:] if len(self.current_session) >= 5 else self.current_session
        
        if not recent_states:
            return 'balanced'  # Default mood
        
        # Calculate aggregate metrics
        avg_cpu = np.mean([s['cpu_percent'] for s in recent_states])
        avg_mem = np.mean([s['memory_percent'] for s in recent_states])
        cpu_variance = np.var([s['cpu_percent'] for s in recent_states])
        
        # Determine mood based on metrics
        if avg_cpu < 30 and avg_mem < 50 and cpu_variance < 100:
            mood = 'calm'  # Low activity, stable
        elif avg_cpu > 70 or avg_mem > 80:
            mood = 'busy'  # High system load
        elif cpu_variance > 200:
            mood = 'distracted'  # Unstable CPU
        else:
            mood = 'balanced'  # Moderate, consistent use
            
        return mood
    
    def generate_sound(self, mood, duration=30):
        """Generate ambient sound based on the current mood"""
        mood_params = self.mood_sounds.get(mood, self.mood_sounds['balanced'])
        scale = mood_params['scale']
        tempo = mood_params['tempo']
        volume = mood_params['volume']
        
        sample_rate = 44100
        combined_audio = np.array([], dtype=np.int16)
        
        # Generate a sequence of notes with mood-specific characteristics
        sequence_length = int(duration / tempo)
        for i in range(sequence_length):
            # Select notes with some randomness but biased by the mood
            note_index = np.random.choice(range(len(scale)))
            note_duration = tempo * (0.8 + np.random.random() * 0.4)
            
            frequency = scale[note_index]
            
            # Generate tone
            t = np.linspace(0, note_duration, int(sample_rate * note_duration), False)
            
            # Use sine waves for tone
            tone = np.sin(frequency * t * 2 * np.pi) * 0.6
            tone += np.sin(frequency*2 * t * 2 * np.pi) * 0.2  # Add higher octave
            
            # Apply fade in/out
            fade_duration = int(sample_rate * 0.2)  # 200ms fade
            if len(tone) > 2 * fade_duration:
                fade_in = np.linspace(0, 1, fade_duration)
                fade_out = np.linspace(1, 0, fade_duration)
                
                tone[:fade_duration] *= fade_in
                tone[-fade_duration:] *= fade_out
            
            # Adjust volume
            tone = tone * volume
            
            # Convert to 16-bit data
            audio = tone * (2**15 - 1) / np.max(np.abs(tone))
            audio = audio.astype(np.int16)
            
            combined_audio = np.append(combined_audio, audio)
            
            # Add a small silence between notes
            silence_duration = tempo * 0.1
            silence = np.zeros(int(sample_rate * silence_duration), dtype=np.int16)
            combined_audio = np.append(combined_audio, silence)
        
        return combined_audio
    
    def visualize_mood(self, mood, audio_data, duration=30):
        """Create a visualization that complements the mood audio"""
        # Create a figure
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, facecolor='black')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        # Create color palette for the current mood
        colors = self.mood_colors.get(mood, self.mood_colors['balanced'])
        cmap = LinearSegmentedColormap.from_list(f"{mood}_cmap", colors, N=100)
        
        # Create background gradient
        gradient = np.random.normal(0.5, 0.1, (20, 20))
        img = ax.imshow(gradient, cmap=cmap, aspect='auto', extent=[-1, 1, -1, 1], alpha=0.8)
        
        # No axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        
        # Set up particle system based on mood
        n_particles = 40
        speed_factor = 0.01
        
        particles = ax.scatter([], [], s=[], c=[], alpha=0.7, cmap=cmap)
        
        # Initialize particle system
        particle_positions = np.random.uniform(-1, 1, (n_particles, 2))
        particle_velocities = np.random.uniform(-speed_factor, speed_factor, (n_particles, 2))
        particle_sizes = np.random.uniform(40, 80, n_particles)
        particle_colors = np.random.uniform(0, 1, n_particles)
        
        # Audio playback
        def play_audio():
            sa.play_buffer(audio_data, 1, 2, 44100).wait_done()
        
        audio_thread = threading.Thread(target=play_audio)
        audio_thread.daemon = True
        
        # Animation update function
        def update(frame):
            nonlocal particle_positions, particle_velocities, particle_sizes, particle_colors
            
            # Update particle positions
            particle_positions += particle_velocities
            
            # Contain particles within boundaries
            out_of_bounds = np.abs(particle_positions) > 1
            particle_velocities[out_of_bounds] *= -0.7
            
            # Add small random movement
            particle_velocities += np.random.uniform(-0.002, 0.002, (n_particles, 2))
            
            # Update particles
            particles.set_offsets(particle_positions)
            particles.set_sizes(particle_sizes)
            particles.set_array(particle_colors)
            
            return particles,
        
        # Start audio and show visualization
        print(f"Starting {mood} mood experience...")
        audio_thread.start()
        
        # Setup animation
        from matplotlib.animation import FuncAnimation
        ani = FuncAnimation(fig, update, frames=300, interval=33.3, blit=True)
        plt.show()
        
        print(f"{mood} mood experience complete.")
    
    def generate_mood_experience(self, duration=30):
        """Generate a complete mood experience based on current system state"""
        # Determine current mood from recent data
        mood = self.determine_mood()
        print(f"Current system mood: {mood}")
        
        # Generate audio for the mood
        audio_data = self.generate_sound(mood, duration)
        
        # Visualize with the audio
        self.visualize_mood(mood, audio_data, duration)
    
    def run_monitoring_session(self, duration_minutes=2, sample_interval=5):
        """Run a monitoring session for the specified duration"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        print(f"Starting Digital Mood Ring monitoring session for {duration_minutes} minutes")
        print("Press Ctrl+C to end the session early")
        
        try:
            while time.time() < end_time:
                # Collect current state
                state = self.collect_system_state()
                
                # Print current state
                print(f"CPU: {state['cpu_percent']}%, Memory: {state['memory_percent']}%")
                
                # Wait for next sample
                time.sleep(sample_interval)
                
            print("Monitoring session complete")
        except KeyboardInterrupt:
            print("Monitoring session stopped by user")

def main():
    mood_ring = DigitalMoodRing()
    
    print("\nDigital Mood Ring")
    print("----------------")
    print("1. Start monitoring session")
    print("2. Generate system mood experience")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        duration = int(input("Enter monitoring duration in minutes (1-5): "))
        duration = max(1, min(5, duration))  # Limit to 1-5 minutes
        mood_ring.run_monitoring_session(duration_minutes=duration)
        main()  # Return to menu
    elif choice == '2':
        print("Collecting system state for mood generation...")
        # Collect some data first if none exists
        if not mood_ring.current_session:
            for _ in range(5):
                mood_ring.collect_system_state()
                time.sleep(2)
        
        duration = int(input("Enter experience duration in seconds (10-30): "))
        duration = max(10, min(30, duration))  # Bound between 10-30 seconds
        mood_ring.generate_mood_experience(duration)
        main()  # Return to menu
    elif choice == '3':
        print("Exiting Digital Mood Ring")
        return
    else:
        print("Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()