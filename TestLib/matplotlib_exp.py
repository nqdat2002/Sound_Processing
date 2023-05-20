# import numpy as np
# import matplotlib.pyplot as plt
#
# # Tạo tín hiệu ban đầu
# n = np.arange(0, 100)
# x = np.sin(2*np.pi*0.05*n)
#
# # Làm chậm tín hiệu
# n0 = -5
# y = np.concatenate([np.zeros(-n0), x[:len(x)+n0]])
#
# # Vẽ đồ thị cho tín hiệu ban đầu và tín hiệu đã làm chậm
# plt.subplot(2, 1, 1)
# plt.plot(n, x)
# plt.title('Tín hiệu ban đầu')
# plt.xlabel('Thời gian (mẫu)')
# plt.ylabel('Amplitude')
#
# plt.subplot(2, 1, 2)
# plt.plot(n, y)
# plt.title('Tín hiệu sau khi làm chậm')
# plt.xlabel('Thời gian (mẫu)')
# plt.ylabel('Amplitude')
#
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Tạo tín hiệu ban đầu
t = np.linspace(0, 2, 200)
x = np.sin(2*np.pi*10*t)

# Làm sớm tín hiệu thêm 5 giây (tương đương với dịch thời gian sang sau 5 giây)
n0 = 2
y = np.concatenate([x[n0:], np.zeros(n0)])

# Vẽ đồ thị cho tín hiệu ban đầu và tín hiệu sau khi làm sớm
t = np.linspace(0, len(x)/100, len(x))
t_fast = np.linspace(0, len(y)/100, len(y))

plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title('Tín hiệu ban đầu')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(t_fast, y)
plt.title('Tín hiệu sau khi làm sớm thêm 5 giây')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
