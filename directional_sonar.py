from impulse_responses import make_room_impulse_responses
from impulse_responses_tester import make_beamformer_test_room_impulse_responses
import sounddevice as sd
from time import sleep
from arlpy import bf, utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# def calculate_delay(array_geometry, sound_speed, theta_source):
#     num_mics = len(array_geometry)
#     delays = np.zeros(num_mics)

#     for i in range(num_mics):
#         mic_position = array_geometry[i]
#         mic_angle = np.arctan2(mic_position[1], mic_position[0])
#         delays[i] = mic_position[0] * np.sin(theta_source - mic_angle) / sound_speed

#     return delays

def calculate_delay(array_geometry, sound_speed, theta_source):
    num_mics = len(array_geometry)
    delays = np.zeros(num_mics)

    for i in range(num_mics):
        mic_position = array_geometry[i]
        mic_distance = np.sqrt(mic_position[0]**2 + mic_position[1]**2)
        mic_angle = np.arctan2(mic_position[1], mic_position[0])
        delays[i] = mic_distance * np.sin(theta_source - mic_angle) / sound_speed

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

# First - test the beamformer on a receiver outside the array
fs = 5000
signals, mic_locs, room = make_beamformer_test_room_impulse_responses()

# angle = np.deg2rad(90)
angles = [0,90,180,270]

# out = delay_and_sum_beamformer(signals, mic_locs, 343, angle)

for angle in np.deg2rad(angles):
    plt.figure()
    out = delay_and_sum_beamformer(signals, mic_locs, 343, angle)
    plt.plot(out)
plt.show()


# print(mic_locs)
# angles = [0,90,180,270]
# angles_test = [[0,0],[90,0], [180,0], [270, 0]]
# delays_test = bf.steering_plane_wave(mic_locs, 343, np.deg2rad(angles_test))
# y = bf.delay_and_sum(signals, 5000, delays_test)

# rsmvals = []
# for i in range(0,np.shape(y)[0]):
#     print(f"{angles_test[i]} - {np.sqrt(np.mean(y[i,:]**2))}")
#     rsmvals.append(np.sqrt(np.mean(y[i,:]**2)))

# plt.figure()
# plt.bar([0,90,180,270], rsmvals, width=2)
# plt.title("RMS uzyskane przez beamformer dla kątów padania")
# plt.ylabel("RMS sygnału")
# plt.xticks([0,90,180,270])
# plt.xlabel("Kąt sterujący")
# plt.grid(True, "both")
# plt.savefig("img_test_beamformer\\test_signal_RMS.png")



#Now test the beamformer on a real scenario

# signals, mic_locs, test_signal = make_room_impulse_responses()
# angles_test = [[0,0],[90,0], [180,0], [270, 0]]
# # angles_test = [[0,0],[0,90], [0,180], [0, 270]]
# delays_test = bf.steering_plane_wave(mic_locs, 343, np.deg2rad(angles_test))
# y = bf.delay_and_sum(signals, 5000, delays_test)

# for i in range(0,np.shape(y)[0]):
#     corr = np.correlate(y[i,:], y[i,:], mode="full")
#     corr_oneside = np.abs(corr[int(np.shape(corr)[0]/2):])

#     peak_indices, peak_dict = scipy.signal.find_peaks(corr_oneside, distance = 100, height=(None, None))
#     peak_heights = peak_dict['peak_heights']
#     highest_peak_index = peak_indices[np.argmax(peak_heights)]
#     second_highest_peak_index = peak_indices[np.argmax(np.delete(peak_heights, np.argmax(peak_heights)))+1] 

#     # peakvals = corr_oneside[peaks]
#     # print(peakvals)
#     # high_to_low = np.argsort(peakvals)
#     # reflection_delay = (peaks[high_to_low[0]]-peaks[high_to_low[1]])/fs
#     reflection_delay = (second_highest_peak_index - highest_peak_index)/fs
#     reflection_distance = 343*reflection_delay/2
#     print(reflection_distance)
#     plt.figure()
#     plt.plot(corr_oneside)
#     plt.plot(peak_indices, peak_heights, "o")
#     plt.plot(highest_peak_index, corr_oneside[highest_peak_index], "x")
#     plt.plot(second_highest_peak_index, corr_oneside[second_highest_peak_index], "x")
#     plt.savefig(f"img\\signal_{angles[i]}_PEAKS_autocorr.png")
#     print(f"{highest_peak_index} - {second_highest_peak_index}")


