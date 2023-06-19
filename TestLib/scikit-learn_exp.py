
# Hãy tải một tập dữ liệu đơn giản có tên là Iris.
# Nó là một bộ dữ liệu về một bông hoa, nó chứa 150 
# quan sát về các phép đo khác nhau của bông hoa. 
# Hãy xem cách tải tập dữ liệu bằng scikit-learning.

# Import scikit learn - khai báo thư viện 
from sklearn import datasets
# Load data - Lấy dữ liệu ra
iris= datasets.load_iris()
# Print shape of data to confirm data is loaded - in dữ liệu mới lấy ra
print(iris.data.shape)


# Bây giờ chúng tôi đã tải dữ liệu, hãy thử học hỏi từ nó và dự đoán 
# trên dữ liệu mới. Với mục đích này, chúng ta phải tạo một công cụ 
# ước tính và sau đó gọi phương thức phù hợp của nó.
from sklearn import svm
from sklearn import datasets
# Load dataset - Lấy dữ liệu ra
iris = datasets.load_iris()
clf = svm.LinearSVC()
# learn from the data 
# dự đoán cho dữ liệu chưa nhìn thấy
clf.fit(iris.data, iris.target)
# predict for unseen data
clf.predict([[ 5.0,  3.6,  1.3,  0.25]])
# Parameters of models can be changed by using the attributes ending with an underscore
#Các tham số của mô hình có thể được thay đổi bằng cách sử dụng các thuộc tính kết thúc bằng dấu gạch dưới
print(clf.coef_ )

# Scikit Tìm hiểu hồi quy tuyến tính
# Tạo các mô hình khác nhau khá đơn giản bằng cách sử dụng 
# scikit-learning. Hãy bắt đầu với một ví dụ đơn giản về hồi quy.

from sklearn import linear_model
reg = linear_model.LinearRegression()
# use it to fit a data  - sử dụng để phù hợp với dữ liệu
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
# Let's look into the fitted data - Hãy xem xét dữ liệu được trang bị
print(reg.coef_)

# k-phân loại hàng xóm gần nhất
# Hãy thử một thuật toán phân loại đơn giản. 
# Bộ phân loại này sử dụng thuật toán dựa trên 
# cây bóng để biểu diễn các mẫu huấn luyện.
from sklearn import datasets

iris = datasets.load_iris()
# Create and fit a nearest-neighbor classifier - khởi tạo và điều chỉnh bộ phân loại lân cận gần nhất
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target)
# Predict and print the result - Dự đoán và in kết quả
result=knn.predict([[0.1, 0.2, 0.3, 0.4]])
print(result)
# kết quả là 0


# K-có nghĩa là phân cụm
# Đây là thuật toán phân cụm đơn giản nhất. Tập hợp được chia 
# thành các cụm 'k' và mỗi quan sát được gán cho một cụm. 
# Điều này được thực hiện lặp đi lặp lại cho đến khi các cụm hội tụ.
# Chúng ta sẽ tạo một mô hình phân cụm như vậy trong chương trình sau:

from sklearn import cluster, datasets
iris = datasets.load_iris()
k=3
k_means = cluster.KMeans(k)
k_means.fit(iris.data)
print( k_means.labels_[::10])
print( iris.target[::10])
