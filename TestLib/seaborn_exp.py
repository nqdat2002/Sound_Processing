import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np
import pandas as pd

# # Exam1
# # Create some data
# rng = np.random.RandomState(0)
# x = np.linspace(0, 10, 500)
# y = np.cumsum(rng.randn(500, 6), 0)

# # Plot the data with Matplotlib defaults
# plt.plot(x, y)
# plt.legend('ABCDEF', ncol=2, loc='upper left')
# plt.show()

# import seaborn as sns
# sns.set()

# plt.plot(x, y)
# plt.legend('ABCDEF', ncol=2, loc='upper left')
# plt.show()

# Exam2
# Thay vì một biểu đồ tần suất, chúng ta có thể có được ước tính
# mượt mà về phân phối bằng cách sử dụng ước tính mật độ nhân,
# điều mà Seaborn thực hiện với sns.kdeplot:
import seaborn as sns
iris = sns.load_dataset("iris")

sns.distplot(iris["sepal_length"])
plt.show()