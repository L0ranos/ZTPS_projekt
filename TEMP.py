import numpy as np
import matplotlib.pyplot as plt

def calculate_delay(array_geometry, sound_speed, theta_source):
    num_mics = len(array_geometry)
    delays = np.zeros(num_mics)

    for i in range(num_mics):
        mic_position = array_geometry[i]
        mic_angle = np.arctan2(mic_position[1], mic_position[0])
        delays[i] = mic_position[0] * np.sin(theta_source - mic_angle) / sound_speed

    return delays

def delay_and_sum_beamformer(microphone_signals, array_geometry, sound_speed, theta_source):
    num_mics = len(microphone_signals)
    num_samples = len(microphone_signals[0])

    # Calculate the time delays for each microphone based on array geometry
    delays = calculate_delay(array_geometry, sound_speed, theta_source)

    # Apply delays to align signals
    aligned_signals = np.zeros_like(microphone_signals)
    for i in range(num_mics):
        delay_samples = int(delays[i] * sound_speed)
        aligned_signals[i] = np.roll(microphone_signals[i], delay_samples)

    # Sum up the aligned signals
    beamformed_signal = np.sum(aligned_signals, axis=0)

    return beamformed_signal

# Parameters
sound_speed = 340.0  # meters per second
theta_source = np.pi/4  # angle of the source in radians

# Array geometry relative to the center of the beamformer
array_geometry = [[0.1, 0], [0, 0.1], [-0.1, 0], [0, -0.1]]

# Generate example microphone signals
num_mics = len(array_geometry)
num_samples = 1000
microphone_signals = np.random.randn(num_mics, num_samples)

# Apply delay-and-sum beamforming
beamformed_signal = delay_and_sum_beamformer(microphone_signals, array_geometry, sound_speed, theta_source)

# Plot the signals and the beamformed signal
plt.figure(figsize=(10, 6))

for i in range(num_mics):
    plt.subplot(num_mics + 1, 1, i + 1)
    plt.plot(microphone_signals[i], label=f'Microphone {i+1}')
    plt.legend()

plt.subplot(num_mics + 1, 1, num_mics + 1)
plt.plot(beamformed_signal, label='Beamformed Signal', color='red')
plt.legend()

plt.tight_layout()
plt.show()
