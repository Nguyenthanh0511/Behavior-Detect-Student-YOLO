from datetime import datetime
import os
import subprocess
from ultralytics import YOLO
import torch

# Hiển thị thông tin GPU hiện tại
output = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(output.stdout)

def get_unique_path(base_path):
    """
    Tạo đường dẫn duy nhất bằng cách thêm phiên bản nếu thư mục đã tồn tại.
    """
    if not os.path.exists(base_path):
        return f"{base_path}_ver{1}"
    version = 2
    while True:
        versioned_path = f"{base_path}_ver{version}"
        if not os.path.exists(versioned_path):
            return versioned_path
        version += 1

# Cấu hình tên model và lưu trữ
current_date = datetime.now().strftime("%d%m%Y_%H%M%S")
nameYoloFamily = 'yolov5s'
ver_dataset = 5
base_save_path = f'/hdd2/minhnv/CodingYOLOv12/StaticModels/{nameYoloFamily}{current_date}_ver-dataset{ver_dataset}/'
unique_save_path = get_unique_path(base_save_path)
os.makedirs(unique_save_path, exist_ok=True)

# Kiểm tra thông tin phiên bản PyTorch và GPU
print(torch.__version__)  
print(torch.cuda.get_device_name(0))  
print(torch.backends.cuda.flash_sdp_enabled())

# Khởi tạo mô hình YOLO từ file cấu hình (ví dụ: "yolov5s.yaml" phải có sẵn)
model = YOLO(f'{nameYoloFamily}.yaml')

# Thay đổi thư mục làm việc sang folder lưu trữ
os.chdir(unique_save_path)

# Đường dẫn file data.yaml của tập dữ liệu oversampled
path_data_yaml = "/hdd2/minhnv/CodingYOLOv12/Dataset/Student-Behavior-Recognition-6_oversampled/data.yaml"

# Cấu hình tham số training
batch_size = 32
num_epochs = 1000

# Huấn luyện mô hình
results = model.train(
    data=path_data_yaml,
    epochs=num_epochs,
    patience=500,             # Số epoch đợi để dừng nếu không cải thiện
    save_period=-1,           # Lưu checkpoint sau mỗi epoch
    save=True,
    optimizer='AdamW',        # Sử dụng optimizer AdamW
    lrf=0.001 * (batch_size / 32),
    batch=batch_size
)

# Lưu mô hình cuối cùng
model_filename = f'{nameYoloFamily}{current_date}.pt'
full_model_path = os.path.join(unique_save_path, model_filename)
if not os.path.exists(full_model_path):
    model.save(full_model_path)
else:
    print(f"Mô hình đã tồn tại tại: {full_model_path}")

print(f"Mô hình đã được lưu tại: {full_model_path}")

# Kiểm tra và in kích thước file mô hình
model_size = os.path.getsize(full_model_path) / (1024 * 1024)  # MB
print(f"Kích thước mô hình: {model_size:.2f} MB")

# Ghi log training
log_path = os.path.join(unique_save_path, f'training_log_{model_filename}.txt')
with open(log_path, 'w', encoding='utf-8') as log_file:
    log_file.write(f"Mô hình: {nameYoloFamily}\n")
    log_file.write(f"Ngày huấn luyện: {current_date}\n")
    log_file.write(f"Số epochs: {num_epochs}\n")
    log_file.write(f"Kích thước mô hình: {model_size:.2f} MB\n")

print(f"Đã ghi log tại: {log_path}")
