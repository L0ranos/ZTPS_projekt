import numpy as np
from scipy.fft import fft, fftfreq, rfft
from scipy.signal.windows import tukey
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import sounddevice as sd
from time import sleep
from pathlib import Path
import pyroomacoustics as pra
from matplotlib.patches import Rectangle

def make_room_impulse_responses():
    fs = 5000
    imp_length = 0.8 # w sekundach

    test_signal = np.random.random(int(fs*imp_length))-0.5

    b, a = butter(5, [1200, 1300], fs=fs, btype='band')
    test_signal = lfilter(b, a, test_signal)

    test_window = tukey(int(fs*imp_length))
    test_signal = test_window*test_signal

    test_spectrum = rfft(test_signal)
    df = fs/test_signal.shape[0]
    fvector = np.arange(0,test_spectrum.shape[0])*df

    # sd.play(test_signal, fs)
    # sleep(imp_length)

    plt.figure()
    plt.plot(test_signal)
    plt.title("Impuls testowy")
    plt.xlabel("Próbka czasowa")
    plt.ylabel("Wartość sygnału")
    plt.grid(True)
    plt.savefig("img\\test_signal.png")

    plt.figure()
    plt.plot(fvector, np.abs(np.real(test_spectrum)))
    plt.title("Widmo impulsu testowego")
    plt.xlabel("Częstotliwość")
    plt.ylabel("Amplituda widma")
    plt.grid(True)
    plt.savefig("img\\test_spectrum.png")


    room_dimensions = [7,9]
    src_position = [2.1, 3.7]
    mic_distance = 0.1 # to zmienić żeby nie było aliasingu
    mic_locs = np.c_[
        [src_position[0]+mic_distance, src_position[1]],  # mic 1
        [src_position[0], src_position[1]-mic_distance],  # mic 2
        [src_position[0]-mic_distance, src_position[1]],  # mic 3
        [src_position[0], src_position[1]+mic_distance],  # mic 4
    ]

    test_room = pra.ShoeBox(
    room_dimensions, fs=fs, materials=pra.Material(0.1), max_order=3)
    test_room.add_source(src_position, signal=test_signal)
    test_room.add_microphone_array(mic_locs)
    test_room.compute_rir()
    test_room.simulate()

    plt.figure(figsize=room_dimensions, dpi=80)
    plt.scatter(mic_locs[0,:], mic_locs[1,:])
    plt.scatter(src_position[0], src_position[1])
    plt.gca().add_patch(Rectangle((0,0),room_dimensions[0], room_dimensions[1], 
                        edgecolor='black',
                        facecolor='none',
                        lw=3,))
    plt.grid(True, "both")
    plt.title("Pokój testowy symulacji")
    plt.savefig("img\\test_room.png")

    plt.figure()
    plt.plot(test_room.rir[1][0])
    plt.title("Odpowiedź impulsowa dla mikrofonu 1")
    plt.xlabel("Czas symulacji")
    plt.ylabel("Amplituda odpowiedzi")
    plt.grid(True, "both")
    plt.savefig("img\\imp_resp_example.png")

    return test_room.mic_array.signals

print(np.shape(make_room_impulse_responses()))