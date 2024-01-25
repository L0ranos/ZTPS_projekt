from impulse_responses import make_room_impulse_responses
from impulse_responses_tester import make_beamformer_test_room_impulse_responses
import sounddevice as sd
from time import sleep
from arlpy import bf, utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
# First - test the beamformer on a receiver outside the array
fs = 5000
signals, mic_locs = make_beamformer_test_room_impulse_responses()

angles = [0,90,180,270]
angles_test = [[0,0],[90,0], [180,0], [270, 0]]
delays_test = bf.steering_plane_wave(mic_locs, 343, np.deg2rad(angles_test))
y = bf.delay_and_sum(signals, 5000, delays_test)

rsmvals = []
for i in range(0,np.shape(y)[0]):
    print(f"{angles_test[i]} - {np.sqrt(np.mean(y[i,:]**2))}")
    rsmvals.append(np.sqrt(np.mean(y[i,:]**2)))

plt.figure()
plt.bar([0,90,180,270], rsmvals, width=2)
plt.title("RMS uzyskane przez beamformer dla kątów padania")
plt.ylabel("RMS sygnału")
plt.xticks([0,90,180,270])
plt.xlabel("Kąt sterujący")
plt.grid(True, "both")
plt.savefig("img_test_beamformer\\test_signal_RMS.png")

#Now test the beamformer on a real scenario

signals, mic_locs = make_room_impulse_responses()
angles_test = [[0,0],[90,0], [180,0], [270, 0]]
delays_test = bf.steering_plane_wave(mic_locs, 343, np.deg2rad(angles_test))
y = bf.delay_and_sum(signals, 5000, delays_test)

for i in range(0,np.shape(y)[0]):
    corr = np.correlate(y[i,:], y[i,:], mode="full")
    corr_oneside = corr[int(np.shape(corr)[0]/2):]
    print(int(np.shape(corr)[0]/2))
    plt.figure()
    plt.plot(corr_oneside)
    plt.savefig("img\\signal_1_autocorr.png")

    peaks, _ = scipy.signal.find_peaks(corr_oneside, distance = 200)
    peakvals = corr_oneside[peaks]
    high_to_low = np.argsort(peakvals)
    reflection_delay = (peaks[high_to_low[0]]-peaks[high_to_low[1]])/fs
    reflection_distance = 343*reflection_delay/2
    print(reflection_distance)
    plt.figure()
    plt.plot(corr_oneside)
    plt.plot(peaks, corr_oneside[peaks], "x")
    plt.savefig(f"img\\signal_{angles[i]}_PEAKS_autocorr.png")


