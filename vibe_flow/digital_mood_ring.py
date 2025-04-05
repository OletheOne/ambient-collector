import time
import os
import psutil
import platform
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import simpleaudio as sa
import threading
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('TkAgg')  # Better for interactive visualization

class DigitalMoodRing:
    def __init__(self):
        self.data_path = "mood_data"
        self.visual_path = "mood_visuals"
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.exists(self.visual_path):
            os.makedirs(self.visual_path)
        
        # Data storage
        self.current_session = []
        self.historic_data = self.load_historic_data()
        
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
    
    def load_historic_data(self):
        """Load all previously saved data sessions"""
        all_data = []
        if os.path.exists(self.data_path):
            for filename in os.listdir(self.data_path):
                if filename.endswith('.json'):
                    with open(os.path.join(self.data_path, filename), 'r') as f:
                        try:
                            data = json.load(f)
                            all_data.extend(data)
                        except json.JSONDecodeError:
                            print(f"Error reading {filename}")
        return all_data
    
    def collect_system_state(self):
        """Collect current system state metrics"""
        timestamp = datetime.now()
        
        # Get CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        per_cpu = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        
        # Get memory metrics
        memory = psutil.virtual_memory()
        mem_percent = memory.percent
        
        # Get disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Get process information
        num_processes = len(list(psutil.process_iter()))
        active_processes = []
        for proc in psutil.process_iter(['name', 'cpu_percent']):
            try:
                proc_info = proc.info
                if proc_info['cpu_percent'] > 1.0:  # Only track active processes
                    active_processes.append({
                        'name': proc_info['name'],
                        'cpu_percent': proc_info['cpu_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Sort and take top processes
        active_processes = sorted(active_processes, key=lambda x: x['cpu_percent'], reverse=True)[:5]
        
        # Build the state object
        state = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'cpu_percent': cpu_percent,
            'cpu_per_core': per_cpu,
            'cpu_freq': cpu_freq,
            'memory_percent': mem_percent,
            'disk_percent': disk_percent,
            'num_processes': num_processes,
            'top_processes': active_processes
        }
        
        self.current_session.append(state)
        return state
    
    def determine_mood(self, recent_states=None):
        """Analyze system state to determine the current 'mood'"""
        if recent_states is None:
            # Use the most recent 5 states or all if fewer
            recent_states = self.current_session[-5:] if len(self.current_session) >= 5 else self.current_session
        
        if not recent_states:
            return 'balanced'  # Default mood
        
        # Calculate aggregate metrics
        avg_cpu = np.mean([s['cpu_percent'] for s in recent_states])
        avg_mem = np.mean([s['memory_percent'] for s in recent_states])
        cpu_variance = np.var([s['cpu_percent'] for s in recent_states])
        
        # Count unique active processes to gauge focus/distraction
        all_processes = []
        for state in recent_states:
            all_processes.extend([p['name'] for p in state.get('top_processes', [])])
        unique_processes = len(set(all_processes))
        
        # Determine mood based on metrics
        if avg_cpu < 30 and avg_mem < 50 and cpu_variance < 100:
            mood = 'calm'  # Low activity, stable
        elif avg_cpu > 70 or avg_mem > 80:
            mood = 'busy'  # High system load
        elif unique_processes > 10 or cpu_variance > 200:
            mood = 'distracted'  # Many different processes active, unstable CPU
        else:
            mood = 'balanced'  # Moderate, consistent use
            
        return mood
    
    def save_session(self):
        """Save the current session data to a file"""
        if not self.current_session:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.data_path, f"mood_session_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(self.current_session, f)
        
        print(f"Session saved to {filename}")
        
        # Add to historic data
        self.historic_data.extend(self.current_session)
    
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
            if mood == 'calm':
                # For calm: longer, lower notes with gentle transitions
                note_index = np.random.choice([0, 1, 2, 4], p=[0.4, 0.3, 0.2, 0.1])
                note_duration = tempo * (1.0 + np.random.random() * 0.5)
            elif mood == 'busy':
                # For busy: shorter, varied notes with quick transitions
                note_index = np.random.choice(range(len(scale)))
                note_duration = tempo * (0.5 + np.random.random() * 0.5)
            elif mood == 'distracted':
                # For distracted: unpredictable pattern
                note_index = np.random.choice(range(len(scale)))
                note_duration = tempo * (0.3 + np.random.random() * 1.0)
            else:  # balanced
                # For balanced: even distribution of notes, regular timing
                note_index = np.random.choice(range(len(scale)))
                note_duration = tempo * (0.8 + np.random.random() * 0.4)
            
            frequency = scale[note_index]
            
            # Generate tone
            t = np.linspace(0, note_duration, int(sample_rate * note_duration), False)
            
            # Use different waveform characteristics for different moods
            if mood == 'calm':
                # Smooth sine waves for calm
                tone = np.sin(frequency * t * 2 * np.pi) * 0.7
                tone += np.sin(frequency/2 * t * 2 * np.pi) * 0.3  # Add subtle lower octave
            elif mood == 'busy':
                # More complex harmonics for busy
                tone = np.sin(frequency * t * 2 * np.pi) * 0.5
                tone += np.sin(frequency*2 * t * 2 * np.pi) * 0.3  # Add higher octave
                tone += np.sin(frequency*3 * t * 2 * np.pi) * 0.1  # Add third harmonic
            elif mood == 'distracted':
                # Slight dissonance for distracted
                tone = np.sin(frequency * t * 2 * np.pi) * 0.6
                tone += np.sin((frequency*1.01) * t * 2 * np.pi) * 0.2  # Slight detuning
                tone += np.sin((frequency*2.02) * t * 2 * np.pi) * 0.1  # Detuned higher octave
            else:  # balanced
                # Clean with gentle harmonics for balanced
                tone = np.sin(frequency * t * 2 * np.pi) * 0.6
                tone += np.sin(frequency*2 * t * 2 * np.pi) * 0.2  # Add higher octave
                tone += np.sin(frequency*1.5 * t * 2 * np.pi) * 0.1  # Add fifth
            
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
        
        # Set up particle system
        if mood == 'calm':
            n_particles = 30
            size_range = (50, 120)
            speed_factor = 0.005
        elif mood == 'busy':
            n_particles = 60
            size_range = (20, 60)
            speed_factor = 0.02
        elif mood == 'distracted':
            n_particles = 45
            size_range = (30, 100)
            speed_factor = 0.015
        else:  # balanced
            n_particles = 40
            size_range = (40, 80)
            speed_factor = 0.01
        
        particles = ax.scatter([], [], s=[], c=[], alpha=0.7, cmap=cmap)
        
        # Initialize particle system
        particle_positions = np.random.uniform(-1, 1, (n_particles, 2))
        particle_velocities = np.random.uniform(-speed_factor, speed_factor, (n_particles, 2))
        particle_sizes = np.random.uniform(size_range[0], size_range[1], n_particles)
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
            
            # Apply mood-specific behavior
            if mood == 'calm':
                # Gentle drifting
                attraction = -particle_positions * 0.001
                particle_velocities += attraction
                particle_velocities *= 0.99  # Damping
            elif mood == 'busy':
                # More energetic movement
                particle_velocities += np.random.uniform(-0.003, 0.003, (n_particles, 2))
                # Occasional bursts
                if frame % 20 == 0:
                    boost_indices = np.random.choice(n_particles, 5)
                    particle_velocities[boost_indices] += np.random.uniform(-0.01, 0.01, (5, 2))
            elif mood == 'distracted':
                # Erratic, changing directions
                if frame % 10 == 0:
                    change_indices = np.random.choice(n_particles, int(n_particles * 0.2))
                    particle_velocities[change_indices] = np.random.uniform(-speed_factor*1.5, speed_factor*1.5, (len(change_indices), 2))
            else:  # balanced
                # Smooth, coordinated movement
                if frame % 30 == 0:
                    # Subtle alignment of movement
                    avg_velocity = np.mean(particle_velocities, axis=0)
                    particle_velocities += avg_velocity * 0.05
            
            # Contain particles within boundaries with soft bounce
            out_of_bounds = np.abs(particle_positions) > 1
            particle_velocities[out_of_bounds] *= -0.7
            
            # Limit maximum velocity
            velocity_magnitude = np.sqrt(np.sum(particle_velocities**2, axis=1))
            too_fast = velocity_magnitude > speed_factor * 2
            if np.any(too_fast):
                particle_velocities[too_fast] *= (speed_factor * 2) / velocity_magnitude[too_fast].reshape(-1, 1)
            
            # Slowly change colors for ambient effect
            particle_colors += np.random.uniform(-0.01, 0.01, n_particles)
            particle_colors = np.clip(particle_colors, 0, 1)
            
            # Update particles
            particles.set_offsets(particle_positions)
            particles.set_sizes(particle_sizes)
            particles.set_array(particle_colors)
            
            # Update background gradient
            if frame % 5 == 0:
                if mood == 'calm':
                    # Calm ripples
                    x = np.linspace(-np.pi, np.pi, 20)
                    y = np.linspace(-np.pi, np.pi, 20)
                    X, Y = np.meshgrid(x, y)
                    wave = 0.05 * np.sin(X * 0.5 + frame/20) * np.sin(Y * 0.5 + frame/25)
                    gradient = 0.5 + wave
                elif mood == 'busy':
                    # More active patterns
                    x = np.linspace(-np.pi, np.pi, 20)
                    y = np.linspace(-np.pi, np.pi, 20)
                    X, Y = np.meshgrid(x, y)
                    wave1 = 0.1 * np.sin(X * 1.0 + frame/10) * np.cos(Y * 0.8 + frame/12)
                    wave2 = 0.08 * np.sin(X * 0.7 + frame/8) * np.cos(Y * 1.2 + frame/10)
                    gradient = 0.5 + wave1 + wave2
                elif mood == 'distracted':
                    # Random shifting patterns
                    gradient = np.random.normal(0.5, 0.1, (20, 20))
                    x = np.linspace(-np.pi, np.pi, 20)
                    y = np.linspace(-np.pi, np.pi, 20)
                    X, Y = np.meshgrid(x, y)
                    noise = 0.2 * np.random.normal(0, 0.1, (20, 20))
                    wave = 0.1 * np.sin(X * 0.8 + frame/15 + noise) * np.cos(Y * 0.9 + frame/18 + noise)
                    gradient += wave
                else:  # balanced
                    # Harmonious patterns
                    x = np.linspace(-np.pi, np.pi, 20)
                    y = np.linspace(-np.pi, np.pi, 20)
                    X, Y = np.meshgrid(x, y)
                    wave = 0.08 * np.sin(X * 0.7 + frame/15) * np.sin(Y * 0.7 + frame/18)
                    gradient = 0.5 + wave
                
                gradient = np.clip(gradient, 0, 1)
                img.set_array(gradient)
            
            return particles, img
        
        # Setup animation
        frames = int(duration * 30)  # 30fps
        ani = FuncAnimation(fig, update, frames=frames, interval=33.3, blit=True)
        
        # Start audio and show visualization
        print(f"Starting {mood} mood experience...")
        audio_thread.start()
        plt.show()
        
        # Wait for audio to complete
        audio_thread.join()
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
    
    def run_monitoring_session(self, duration_minutes=10, sample_interval=30):
        """Run a monitoring session for the specified duration"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        print(f"Starting Digital Mood Ring monitoring session for {duration_minutes} minutes")
        print("Press Ctrl+C to end the session early")
        
        try:
            while time.time() < end_time:
                # Collect current state
                state = self.collect_system_state()
                
                # Print current mood every few samples
                if len(self.current_session) % 5 == 0:
                    mood = self.determine_mood()
                    print(f"Current system mood: {mood}")
                    print(f"CPU: {state['cpu_percent']}%, Memory: {state['memory_percent']}%")
                    if state['top_processes']:
                        print(f"Top process: {state['top_processes'][0]['name']}")
                
                # Wait for next sample
                time.sleep(sample_interval)
                
            print("Monitoring session complete")
        except KeyboardInterrupt:
            print("Monitoring session stopped by user")
        
        # Save the session data
        self.save_session()
    
    def generate_insights(self):
        """Analyze historical data to generate insights about system usage patterns"""
        if not self.historic_data:
            print("No historical data available for analysis")
            return
        
        # Convert to pandas DataFrame for easier analysis
        data = []
        for entry in self.historic_data:
            data.append({
                'timestamp': entry['timestamp'],
                'cpu_percent': entry['cpu_percent'],
                'memory_percent': entry['memory_percent'],
                'num_processes': entry['num_processes']
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Generate time-based patterns
        hour_avg = df.groupby('hour').mean()
        day_avg = df.groupby('day_of_week').mean()
        
        # Create visualization of patterns
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot hourly patterns
        hour_avg.plot(ax=ax1)
        ax1.set_title('System Resource Usage by Hour of Day')
        ax1.set_xticks(range(24))
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Resource Usage (%)')
        ax1.legend()
        
        # Plot daily patterns
        day_avg.plot(ax=ax2)
        ax2.set_title('System Resource Usage by Day of Week')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Resource Usage (%)')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the insights visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        insight_file = os.path.join(self.visual_path, f"insights_{timestamp}.png")
        plt.savefig(insight_file)
        plt.close()
        
        print(f"Usage pattern insights saved to {insight_file}")
        
        # Determine peak productivity times
        cpu_peak_hour = hour_avg['cpu_percent'].idxmax()
        mem_peak_hour = hour_avg['memory_percent'].idxmax()
        
        cpu_peak_day = day_avg['cpu_percent'].idxmax()
        mem_peak_day = day_avg['memory_percent'].idxmax()
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        print("\n=== System Usage Insights ===")
        print(f"Peak CPU usage typically occurs at {cpu_peak_hour}:00 on {days[cpu_peak_day]}")
        print(f"Peak memory usage typically occurs at {mem_peak_hour}:00 on {days[mem_peak_day]}")
        
        # Calculate system mood distribution
        moods = []
        for i in range(0, len(self.historic_data), 5):
            data_window = self.historic_data[i:i+5]
            if data_window:
                mood = self.determine_mood(data_window)
                moods.append(mood)
        
        if moods:
            mood_counts = {}
            for mood in moods:
                mood_counts[mood] = mood_counts.get(mood, 0) + 1
            
            total = sum(mood_counts.values())
            print("\nSystem Mood Distribution:")
            for mood, count in mood_counts.items():
                percentage = (count / total) * 100
                print(f"  {mood.title()}: {percentage:.1f}%")

# Main execution function
def main():
    mood_ring = DigitalMoodRing()
    
    print("\nDigital Mood Ring")
    print("----------------")
    print("1. Start monitoring session")
    print("2. Generate system mood experience")
    print("3. Analyze system patterns")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        duration = int(input("Enter monitoring duration in minutes: "))
        mood_ring.run_monitoring_session(duration_minutes=duration)
    elif choice == '2':
        print("Collecting system state for mood generation...")
        # Collect some data first if none exists
        if not mood_ring.current_session:
            for _ in range(5):
                mood_ring.collect_system_state()
                time.sleep(2)
        
        duration = int(input("Enter experience duration in seconds (15-60): "))
        duration = max(15, min(60, duration))  # Bound between 15-60 seconds
        mood_ring.generate_mood_experience(duration)
    elif choice == '3':
        if not mood_ring.historic_data and not mood_ring.current_session:
            print("No data available for analysis. Please run a monitoring session first.")
        else:
            if mood_ring.current_session:
                mood_ring.save_session()
            mood_ring.generate_insights()
    elif choice == '4':
        print("Exiting Digital Mood Ring")
        return
    else:
        print("Invalid choice. Please try again.")
    
    # Recursive call to show menu again
    main()

if __name__ == "__main__":
    main()