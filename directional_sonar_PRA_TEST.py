from impulse_responses import make_room_impulse_responses
from impulse_responses_tester import make_beamformer_test_room_impulse_responses
import sounddevice as sd
from time import sleep
from arlpy import bf, utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.fft import fft, fftfreq, rfft
from scipy.signal.windows import tukey
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from time import sleep
from pathlib import Path
import pyroomacoustics as pra
from matplotlib.patches import Rectangle
from arlpy import bf, utils
# First - test the beamformer on a receiver outside the array
fs = 5000
# signals, mic_locs, room = make_beamformer_test_room_impulse_responses()
angles = [0,90,180,270]
for angle in angles:

    imp_length = 0.8 # w sekundach
    test_signal = np.random.random(int(fs*imp_length))-0.5
    b, a = butter(5, [700, 800], fs=fs, btype='band')
    test_signal = lfilter(b, a, test_signal)
    test_window = tukey(int(fs*imp_length))
    test_signal = test_window*test_signal
    test_spectrum = rfft(test_signal)
    df = fs/test_signal.shape[0]
    fvector = np.arange(0,test_spectrum.shape[0])*df

    room_dimensions = [7,9]
    src_position = [2.1, 3.7]
    outside_src_1 = [2.1,7]
    mic_distance = 0.1 # to zmienić żeby nie było aliasingu
    mic_locs_abs = np.c_[
    [src_position[0]+mic_distance, src_position[1]],  # mic 1
    [src_position[0], src_position[1]-mic_distance],  # mic 2
    [src_position[0]-mic_distance, src_position[1]],  # mic 3
    [src_position[0], src_position[1]+mic_distance],  # mic 4
    ]

    mic_locs_rel = np.asarray([
    [0+mic_distance, 0],  # mic 1
    [0, 0-mic_distance],  # mic 2
    [0-mic_distance, 0],  # mic 3
    [0, 0+mic_distance],  # mic 4
    ])

    Lg_t = 0.100                
    Lg = np.ceil(Lg_t*fs)       
    room_bf = pra.ShoeBox([20,40], fs=fs, max_order=12)
    source = np.array([10, 20])
    room_bf.add_source(source, delay=0., signal=test_signal)
    center = [10, 20]; radius = 0.1
    fft_len = 512
    echo = pra.circular_2D_array(center=center, M=12, phi0=0, radius=radius)
    mics = pra.Beamformer(echo, room_bf.fs, N=fft_len, Lg=Lg)
    room_bf.add_microphone_array(mics)

# Compute DAS weights
# mics.rake_delay_and_sum_weights(room_bf.sources[0][:1])

    mics.far_field_weights(np.deg2rad(angle))
    room_bf.compute_rir()
    room_bf.simulate()

    signal_das = mics.process(FD=False)
    sd.play(signal_das/10, fs)

    fig, ax = room_bf.plot(freq=[800], img_order=0)
    ax.legend(['800'])
    plt.title(f"Charakterystyka beamformera dla kąta {angle} stopni")
    plt.tight_layout()
    plt.savefig(f"img\\signal_{angle}_room_beamformer.png")


    corr = np.correlate(signal_das, signal_das, mode="full")
    corr_oneside = np.abs(corr[int(np.shape(corr)[0]/2):])

    peak_indices, peak_dict = scipy.signal.find_peaks(corr_oneside, distance = 100, height=(None, None))
    peak_heights = peak_dict['peak_heights']
    highest_peak_index = peak_indices[np.argmax(peak_heights)]
    second_highest_peak_index = peak_indices[np.argmax(np.delete(peak_heights, np.argmax(peak_heights)))+1] 

    reflection_delay = (second_highest_peak_index - highest_peak_index)/fs
    reflection_distance = 343*reflection_delay/2
    print(reflection_distance)
    plt.figure()
    plt.plot(corr_oneside)
    plt.plot(peak_indices, peak_heights, "o")
    plt.plot(highest_peak_index, corr_oneside[highest_peak_index], "x")
    plt.plot(second_highest_peak_index, corr_oneside[second_highest_peak_index], "x")
    plt.legend(["Autokorelacja", "Wykryte piki", "Najwyższy pik", "Drugi najwyższy pik"])
    plt.tight_layout()
    plt.savefig(f"img\\signal_{angle}_PEAKS_autocorr.png")
    print(f"{highest_peak_index} - {second_highest_peak_index}")
