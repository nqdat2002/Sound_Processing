from numpy import arccos, dot
import  numpy as np
from numpy.linalg import norm

# test về vector trong Numpy: sử dụng phương thức khởi tạo 2 vector có kích thước 1*3
# sau đó sử dụng phương thức dot: để tính tích vô hướng của 2 vector này
# phương thức norm để tính độ dài vector
# arccos để tính 2 góc của 2 vector dựa trên công thức v⋅w=|v|*|w|*cos(v, w),
# góc phi được tính và hiển thị kết quả, giá trị trả về thuộc khoảng từ [-1, 1]
# numpy.arccos(x, /, out=None, *, where=True)
#   tham số thứ nhất: là giá trị truyền vào,
#   tham số thứ hai: là vị trí kết quả lưu trữ, có thể tùy chọn
#   tham số thứ ba: là đại diện đối số từ từ khóa khác.
#   tham số thứ tư: Đây là điều kiện mà đầu vào được quảng bá.
#       Tại một vị trí nhất định có điều kiện này True, mảng kết quả sẽ được đặt thành kết quả ufunc.
#       Nếu không, mảng kết quả sẽ giữ nguyên giá trị ban đầu của nó. Đây là một giá trị tham số tùy chọn.
#   Chúng ta cũng có thể viết tắt:
        # /
        # arccos(x,

v = np.array([[10, 9, 3]]) # khởi tạo 2 vector
w = np.array([[2, 5, 12]])
phi = \
    arccos(dot(v, w.T)/(norm(v)*norm(w))) # tính góc giữa 2 vector.
print(phi) # hiển thị kết quả góc 2 vector.