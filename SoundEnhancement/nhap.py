# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# Fs = 1000  # Tần số lấy mẫu
# f = 10  # Tần số của tín hiệu
# t = np.arange(0, 1, 1/Fs)
# x = np.sin(2*np.pi*f*t)
#
# # Thiết kế bộ lọc thông thấp
# fc = 50  # Tần số cắt
# b = signal.firwin(101, fc/(Fs/2), window='hamming')
# y = signal.convolve(x, b, mode='same')
#
# # Tìm đỉnh trung tâm cao nhất thứ cấp của tần số
# frequencies, power_spectrum = signal.welch(y, fs=Fs, nperseg=256)
# peaks, _ = signal.find_peaks(power_spectrum, height=0)
# idx = np.argmax(power_spectrum[peaks])
# second_highest_peak = peaks[idx]
#
# # Hiển thị kết quả trên đồ thị
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
#
# # Đồ thị tín hiệu ban đầu và sau khi lọc
# ax1.plot(t, x, label='Original Signal')
# ax1.plot(t, y, label='Filtered Signal')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Amplitude')
# ax1.legend()
#
# # Đồ thị biên độ phổ
# ax2.plot(frequencies, power_spectrum)
# ax2.plot(frequencies[second_highest_peak], power_spectrum[second_highest_peak], 'r.', markersize=10, label='Second highest peak')
# ax2.set_xlabel('Frequency (Hz)')
# ax2.set_ylabel('Power/Frequency (dB/Hz)')
# ax2.legend()
#
# plt.show()
#
#
#
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# N = 256
# t = np.arange(N)
#
# m = 4
# nu = float(m)/N
# f = np.sin(2*np.pi*nu*t)
# ft = np.fft.fft(f)
# freq = np.fft.fftfreq(N)
# plt.plot(freq, ft.real**2 + ft.imag**2)
# plt.show()
#
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Number of sample points
# N = 1000
#
# # Sample spacing
# T = 1.0 / 800.0     # f = 800 Hz
#
# # Create a signal
# x = np.linspace(0.0, N*T, N)
# t0 = np.pi/6   # non-zero phase of the second sine
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(200.0 * 2.0*np.pi*x + t0)
# yf = np.fft.fft(y) # to normalize use norm='ortho' as an additional argument
#
# # Where is a 200 Hz frequency in the results?
# freq = np.fft.fftfreq(x.size, d=T)
# # index, = np.where(np.isclose(freq, 200, atol=1/(T*N)))
#
# # Get magnitude and phase
# magnitude = np.abs(yf)
# phase = np.angle(yf)
# print("Magnitude:", magnitude, ", phase:", phase)
#
# # Plot a spectrum
# plt.plot(freq[0:N//2], 2/N*np.abs(yf[0:N//2]), label='amplitude spectrum')   # in a conventional form
# plt.plot(freq[0:N//2], np.angle(yf[0:N//2]), label='phase spectrum')
# plt.plot(freq[:N//2], magnitude[:N//2])
# plt.legend()
# plt.grid()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Tín hiệu mẫu
Fs = 1000
t = np.arange(0, 1, 1/Fs)
x = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*150*t)

# Tín hiệu trong miền tần số
X = np.fft.fft(x)
freq = np.fft.fftfreq(len(x), 1/Fs)

# Thiết kế bộ lọc thông cao
cutoff = 100 # Tần số cắt
H = np.ones(len(freq))
H[freq < -cutoff] = 0
H[freq > cutoff] = 0

# Lọc tín hiệu
Y = X * H
y = np.real(np.fft.ifft(Y))

# Hiển thị kết quả
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t, x)
plt.title('Tín hiệu gốc')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')

plt.subplot(2,1,2)
plt.plot(t, y)
plt.title('Tín hiệu sau khi lọc')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')

plt.tight_layout()
plt.show()