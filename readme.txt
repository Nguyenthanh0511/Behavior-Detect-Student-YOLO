# Hệ Thống Phát Hiện Sinh Viên Trong Lớp Học

Dự án này phát triển hệ thống phát hiện đối tượng sinh viên trong lớp học, góp phần nâng cao hiệu quả quản lý lớp học.

## Tổng Quan

Hệ thống sử dụng mô hình YOLOv12 phiên bản small để nhận diện sinh viên trong lớp học qua camera, kết hợp với giao diện web để hiển thị kết quả và quản lý.

## Thành Phần Chính

### Camera Module
- Thu thập hình ảnh từ lớp học
- Truyền dữ liệu qua đường stream

### Flask API
- Xử lý các yêu cầu từ Web UI
- Xử lý hình ảnh/video từ camera
- Áp dụng mô hình YOLOv12 để phát hiện sinh viên
- Lưu trữ kết quả và phân tích dữ liệu

### Web UI
- Hiển thị video trực tiếp từ camera
- Hiển thị kết quả phát hiện đối tượng
- Cung cấp giao diện quản lý và thống kê

## Hướng Dẫn Cài Đặt

1. Cài đặt các thư viện Python cần thiết:
   ```
   pip install -r requirements.txt
   ```

2. Cấu hình camera:
   - Lấy địa chỉ IP của camera
   - Thêm đường stream từ IP camera (xem thêm trong file route.py để hiểu cách cấu hình)

3. Khởi động máy chủ:
   ```
   python run.py
   ```

4. Truy cập giao diện web tại địa chỉ server đã cấu hình

## Cấu Trúc Dự Án

- `app/` - Chứa mã nguồn chính của ứng dụng web
- `app/templates/index.html` - Giao diện người dùng (điều chỉnh ở đây)
- `app/routes.py` - Định tuyến và xử lý yêu cầu API (điều chỉnh ở đây)
- Các thư mục ngoài folder server chứa mã nguồn dùng để huấn luyện mô hình, đánh giá và thống kê

## Dữ Liệu

Dữ liệu huấn luyện là tập dữ liệu riêng được thu thập bởi nhóm phát triển và hiện không được công khai. Nếu cần dữ liệu cho mục đích nghiên cứu, vui lòng liên hệ với tác giả.

## Liên Hệ

[Thông tin liên hệ của bạn]

---

*Lưu ý: Đọc thêm file route.py để hiểu chi tiết về cách cấu hình đường stream camera.*