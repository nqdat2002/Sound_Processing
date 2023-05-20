import wave
import numpy as np
from matplotlib import pyplot as plt

def showGraph(filePath):
    file_name = filePath
    with wave.open(file_name, "rb") as wave_file:
        sample_rate = wave_file.getframerate()
        num_frames = wave_file.getnframes()
        num_channels = wave_file.getnchannels()
        sample_width = wave_file.getsampwidth()
        raw_data = wave_file.readframes(num_frames)

    # Biên độ theo thời gian
    data = np.frombuffer(raw_data, dtype=np.int16)
    times = np.linspace(0, len(data) / sample_rate, num=len(data))
    plt.subplot(4, 1, 1)
    plt.plot(times, data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Time domain waveform")
    plt.grid()

def echo(filename, time_delay):
    wav_obj = wave.open(str(filename), 'rb')
    sample_freq = wav_obj.getframerate()  # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    output_audio = np.zeros(len(signal_array))
    output_delay = time_delay * sample_freq
    for count, e in enumerate(signal_array):
        output_audio[count] = e + signal_array[count - int(output_delay)]
    signal_array = output_audio

    filtered_audio_frames = signal_array.astype(np.int16).tobytes()
    filtered_audio_file = wave.open('echo' + filename, 'wb')
    filtered_audio_file.setnchannels(n_channels)
    filtered_audio_file.setsampwidth(wav_obj.getsampwidth())
    filtered_audio_file.setframerate(sample_freq)
    filtered_audio_file.setnframes(n_samples)
    filtered_audio_file.writeframes(filtered_audio_frames)
    filtered_audio_file.close()

def faded(filename, alpha, time, type='in'):
    wav_obj = wave.open(str(filename), 'rb')
    sample_freq = wav_obj.getframerate()  # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    if type == 'in':
        fade_samples = time * sample_freq
        fade_in_curve = alpha * np.linspace(0, 1, fade_samples)
        signa1 = np.array(signal_array[0:fade_samples])
        signa2 = np.array(signal_array[fade_samples:])
        one = np.ones(len(signal_array) - len(fade_in_curve))
        signa3 = signa2 * one
        signa4 = signa1 * fade_in_curve
        signal_array = np.concatenate([signa4, signa3])
    else:
        fade_samples = time * sample_freq
        fade_in_curve = alpha * np.linspace(1, 0, fade_samples)
        signa1 = np.array(signal_array[0:len(signal_array) - fade_samples])
        signa2 = np.array(signal_array[len(signal_array) - fade_samples:])
        one = np.ones(len(signal_array) - len(fade_in_curve))
        signa3 = signa1 * one
        signa4 = signa2 * fade_in_curve
        signal_array = np.concatenate([signa3, signa4])

    filtered_audio_frames = signal_array.astype(np.int16).tobytes()
    filtered_audio_file = wave.open('faded_' + type + filename, 'wb')
    filtered_audio_file.setnchannels(n_channels)
    filtered_audio_file.setsampwidth(wav_obj.getsampwidth())
    filtered_audio_file.setframerate(sample_freq)
    filtered_audio_file.setnframes(n_samples)
    filtered_audio_file.writeframes(filtered_audio_frames)
    filtered_audio_file.close()

def reversing(filename):
    wav_obj = wave.open(str(filename), 'rb')
    sample_freq = wav_obj.getframerate()  # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    signal_array = signal_array[::-1]

    filtered_audio_frames = signal_array.astype(np.int16).tobytes()
    filtered_audio_file = wave.open('reversing' + filename, 'wb')
    filtered_audio_file.setnchannels(n_channels)
    filtered_audio_file.setsampwidth(wav_obj.getsampwidth())
    filtered_audio_file.setframerate(sample_freq)
    filtered_audio_file.setnframes(n_samples)
    filtered_audio_file.writeframes(filtered_audio_frames)
    filtered_audio_file.close()

def modulation(filename, alpha, wc = 20):
    wav_obj = wave.open(str(filename), 'rb')
    sample_freq = wav_obj.getframerate()  # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    lm = np.array([1 + alpha * np.cos(wc * i) for i in range(len(signal_array))])
    signal_array = signal_array * lm

    filtered_audio_frames = signal_array.astype(np.int16).tobytes()
    filtered_audio_file = wave.open('modulation' + filename, 'wb')
    filtered_audio_file.setnchannels(n_channels)
    filtered_audio_file.setsampwidth(wav_obj.getsampwidth())
    filtered_audio_file.setframerate(sample_freq)
    filtered_audio_file.setnframes(n_samples)
    filtered_audio_file.writeframes(filtered_audio_frames)
    filtered_audio_file.close()

def main():
    # echo('sang-amthanh2.wav', time_delay=2)
    reversing('sang-amthanh2.wav')
    modulation('sang-amthanh2.wav', alpha=0.5, wc = 20)
    faded('sang-amthanh2.wav', alpha=1, time=2, type='in')

if __name__ == '__main__':
    main()