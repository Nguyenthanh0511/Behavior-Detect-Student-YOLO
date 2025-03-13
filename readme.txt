# Hệ Thống Phát Hiện Sinh Viên Trong Lớp Học

<p align="center">
  <img src="/api/placeholder/800/400" alt="Student Detection System" />
</p>

Dự án này phát triển hệ thống phát hiện đối tượng sinh viên trong lớp học, góp phần nâng cao hiệu quả quản lý lớp học.

## 📋 Tổng Quan

Hệ thống sử dụng mô hình YOLOv12 phiên bản small để nhận diện sinh viên trong lớp học qua camera, kết hợp với giao diện web để hiển thị kết quả và quản lý.

## 🔍 Thành Phần Chính

### 📷 Camera Module
- Thu thập hình ảnh từ lớp học
- Truyền dữ liệu qua đường stream

### 🖥️ Flask API
- Xử lý các yêu cầu từ Web UI
- Xử lý hình ảnh/video từ camera
- Áp dụng mô hình YOLOv12 để phát hiện sinh viên
- Lưu trữ kết quả và phân tích dữ liệu

### 🌐 Web UI
- Hiển thị video trực tiếp từ camera
- Hiển thị kết quả phát hiện đối tượng
- Cung cấp giao diện quản lý và thống kê

## ⚙️ Hướng Dẫn Cài Đặt

1. Clone repository này về máy:
   ```bash
   git clone https://github.com/your-username/student-detection-system.git
   cd student-detection-system
   ```

2. Cài đặt các thư viện Python cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

3. Cấu hình camera:
   - Lấy địa chỉ IP của camera
   - Thêm đường stream từ IP camera (xem thêm trong file `route.py` để hiểu cách cấu hình)

4. Khởi động máy chủ:
   ```bash
   python run.py
   ```

5. Truy cập giao diện web tại địa chỉ server đã cấu hình

## 📁 Cấu Trúc Dự Án

```
student-detection-system/
├── app/                     # Thư mục chính của ứng dụng
│   ├── __init__.py          # Khởi tạo ứng dụng Flask
│   ├── routes.py            # Định tuyến API và xử lý yêu cầu
│   ├── models/              # Mô hình YOLOv12
│   ├── static/              # CSS, JavaScript, hình ảnh
│   └── templates/           # HTML templates
│       └── index.html       # Giao diện người dùng chính
├── training/                # Mã nguồn huấn luyện mô hình
├── evaluation/              # Mã nguồn đánh giá mô hình
├── statistics/              # Mã nguồn phân tích thống kê
├── run.py                   # Script khởi động ứng dụng
└── requirements.txt         # Danh sách thư viện cần thiết
```

## 📊 Dữ Liệu

Dữ liệu huấn luyện là tập dữ liệu riêng được thu thập bởi nhóm phát triển và hiện không được công khai. Nếu cần dữ liệu cho mục đích nghiên cứu, vui lòng liên hệ với tác giả.

## 🛠️ Công Nghệ Sử Dụng

- **Deep Learning**: YOLOv12, PyTorch
- **Backend**: Flask, OpenCV
- **Frontend**: HTML, CSS, JavaScript
- **Phân tích dữ liệu**: NumPy, Pandas, Matplotlib

## 📝 Liên Hệ

[Thông tin liên hệ của bạn]

## 📜 Giấy Phép

[Thông tin giấy phép]

---

*Lưu ý: Đọc thêm file `route.py` để hiểu chi tiết về cách cấu hình đường stream camera.*
