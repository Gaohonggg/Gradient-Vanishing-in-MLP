Tìm hiểu các cách có thể cải thiện mô hình qua các kĩ thuật mà không cần giải bài toán bằng cách tìm một mô hình mới hoàn toàn.

Time_1.py: Là đoạn code MLP ban đầu, sau khi chạy dễ nhận thấy được mô hình không học được vì vấn đề Gradient-Vanishing.

Time_2.py: Cải thiện mô hình bằng phương pháp Khởi tạo các trọng số tốt ở các lớp layer, cụ thể ở đây là khởi tạo với mean = 0 
           và std = 1.
           Qua bước này thấy được mô hình đã có khả năng học so với trước đó( accuracy = 74% ).

Time_3.py: Sử dụng hàm activattion ReLU thay thế cho Sigmoid và thay đổi std = 0.05.
           Sau bước này mô hình đã có sự cải thiện hơn( accuracy = 81% ), tăng 7%.

Time_4.py: Thay thế hàm tối ưu từ optim.SGD thành optim.Adam. 
           Mô hình tiếp tục tăng thêm 6% (accuracy = 87% ).

Time_5.py: Sử dụng thêm lớp BatchNormalize cho các layer để tăng tốc độ hội tụ và cải thiện khả năng tổng quát của mô hình.
           Khi sử dụng tại activation là Sigmoid thì accuracy không cải thiện so với Time_4.
           Khi sử dụng tiếp ReLU thì accuracy có phần giảm nhẹ so với trước.

Time_6.py: Kỹ năng Skip-Connection được giới thiệu trong mạng ResNet, giúp cải thiện tốc độ hội tụ và không bị mất thông tin 
           từ các tầng trước đó.
           Khi sử dụng tiếp activation Sigmoid và BatchNormalize thì accuracy hầu như không có cải thiện.
           Khi dùng Sigmoid và khong dùng BatchNormalize thì accuracy có tăng thêm 1%.
           Khi dùng ReLU thì accuracy cải thiện được 2%

Time_7.py: Fine-tune lại các giai đoạn, việc huấn luyện toàn bộ các lớp có thể dẫn đến hiệu suất thấp do vấn đề Vanishing 
           Gradient. Bằng cách chỉ huấn luyện một số lớp cụ thể, mô hình có thể tập trung vào việc học các đặc trưng quan 
           trọng hơn mà không bị ảnh hưởng bởi các lớp sâu hơn. Kỹ thuật này được triển khai bằng cách xây dựng các mô hình 
           nhỏ hơn tương ứng với từng số lượng lớp cần huấn luyện, sau đó tăng dần các lớp được tham gia huấn luyện.

Time_8.py: Gradient Normalization: Là một kỹ thuật với ý tưởng chuẩn hóa gradient trong quá trình lan truyền ngược. Kỹ thuật 
           này đảm bảo gradient được duy trì trong một phạm vi hợp lý, tránh việc chúng trở nên quá nhỏ hoặc quá lớn, từ đó 
           giúp cải thiện quá trình học của các lớp sâu hơn trong mạng. Tại đây, chúng ta cài đặt một lớp 
           GradientNormalizationLayer, sử dụng cơ chế autograd của PyTorch để chuẩn hóa gradient trong giai đoạn lan truyền 
           ngược. Cụ thể, gradient được điều chỉnh bằng cách chuẩn hóa theo trung bình và độ lệch chuẩn của chúng, đảm bảo 
           các giá trị gradient không bị triệt tiêu hoặc phóng đại.

Qua các bước sẽ thấy được mô hình đã được cải tiến rất nhiều, Từ không học được đã trở thành mô hình tốt => Không phải lúc nào cứ tìm một mô hình mới cũng tốt, có thể dựa vào các cách để cải thiện chính mô hình đó để thử. Vì ngay ở tập này, việc huấn luyện chỉ dựa vào các lớp MLP ban đầu và cải thiện lên nhờ các kĩ thuật chứ không dựa vào một mô hình mới hoàn toàn.
