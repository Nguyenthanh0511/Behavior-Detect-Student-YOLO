data_yaml_path = "/hdd2/minhnv/CodingYOLOv12/Dataset/Student-Behavior-Recognition-6_oversampled/data.yaml"
from datetime import datetime
import os
import subprocess
# Chạy lệnh nvidia-smi và in kết quả
output = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(output.stdout)
def get_unique_path(base_path):
    """
    Tạo đường dẫn duy nhất bằng cách thêm phiên bản nếu thư mục đã tồn tại

    Args:
        base_path (str): Đường dẫn gốc muốn tạo

    Returns:
        str: Đường dẫn duy nhất
    """
    # Nếu thư mục chưa tồn tại, trả về ngay
    if not os.path.exists(base_path):
        return f"{base_path}_ver{1}"

    # Nếu đã tồn tại, tìm phiên bản tiếp theo
    version = 1
    while True:
        versioned_path = f"{base_path}_ver{version}"
        if not os.path.exists(versioned_path):
            return versioned_path
        version += 1
        

# Tạo đường dẫn lưu trữ với thư mục theo ngày
current_date = datetime.now().strftime("%d%m%Y_%H%M%S")
nameYoloFamily = 'yolov12s'
### base
ver_dataset = 6
base_save_path = f'/hdd2/minhnv/CodingYOLOv12/StaticModels/{nameYoloFamily}{current_date}_ver-dataset{ver_dataset}/'

# Lấy đường dẫn duy nhất ----------------------------------Đường dẫn này cũng là quan trọng. Vì các báo cáo dưới đều càn đến
unique_save_path = get_unique_path(base_save_path)
# Tạo thư mục
os.makedirs(unique_save_path, exist_ok=True)


import os
import shutil
from datetime import datetime
from ultralytics import YOLO
import torch
print(torch.__version__)  # Check PyTorch version
print(torch.cuda.get_device_name(0))  # Check GPU model
print(torch.backends.cuda.flash_sdp_enabled())  # Check if FlashAttention is available


#-----------------Train
import pandas as pd
# Tạo đường dẫn lưu trữ với thư mục theo ngày
nameYoloFamily = 'yolov12s'

# Lấy đường dẫn duy nhất ----------------------------------Đường dẫn này cũng là quan trọng. Vì các báo cáo dưới đều càn đến
unique_save_path = get_unique_path(base_save_path)
# Tạo thư mục
os.makedirs(unique_save_path, exist_ok=True)

# Tạo mô hình và huấn luyện
model_filename = f'{nameYoloFamily}{current_date}.pt'
full_model_path = os.path.join(unique_save_path, model_filename)
# Tạo mô hình YOLO và cấu hình
model = YOLO(f'{nameYoloFamily}.yaml')

# Thay đổi thư mục làm việc
# %cd "{unique_save_path}"
os.chdir(unique_save_path)
# Dữ liệu
path_data_yaml = "/hdd2/minhnv/CodingYOLOv12/Dataset/Student-Behavior-Recognition-6/data.yaml"


#------------------ Traning model
'''
Giả sử bạn bắt đầu với batch size = 32 và learning rate = 0.001.

Nếu bạn tăng batch size lên 64, bạn có thể tăng learning rate lên 0.001 * (64/32) = 0.002.

Nếu bạn giảm batch size xuống 16, bạn có thể giảm learning rate xuống 0.001 * (16/32) = 0.0005.
'''
batch_size = 32
num_epochs = 1000
# Khởi tạo log
log_data = []

results = model.train(
    data=path_data_yaml,
    epochs=num_epochs,
    patience=500,             # Số epoch đợi để dừng nếu không cải thiện
    save_period=-1,           # Lưu checkpoint sau mỗi epoch
    save=True,
    optimizer='AdamW',        # Lựa chọn optimizer
    lrf=0.001 * (batch_size / 32),
    batch=batch_size,
    # weight_decay=1e-4,        # Ví dụ: dùng weight decay là 1e-4
    # dropout=0.3              # Ví dụ: dùng dropout rate 30%
)

# Lưu mô hình cuối cùng
if not os.path.exists(full_model_path):
    model.save(full_model_path)
else:
  print(f"Mô hình đã tồn tại tại: {full_model_path}")

# In thông tin về mô hình đã lưu
print(f"Mô hình đã được lưu tại: {full_model_path}")

# Kiểm tra và in kích thước file
model_size = os.path.getsize(full_model_path) / (1024 * 1024)  # Chuyển sang MB
print(f"Kích thước mô hình: {model_size:.2f} MB")

# Đảm bảo sử dụng UTF-8 encoding
log_path = os.path.join(unique_save_path, f'training_log_{model_filename}.txt')
with open(log_path, 'w', encoding='utf-8') as log_file:
    log_file.write(f"Mô hình: {nameYoloFamily}\n")
    log_file.write(f"Ngày huấn luyện: {current_date}\n")
    log_file.write(f"Số epochs: {num_epochs} \n")
    log_file.write(f"Kích thước mô hình: {model_size:.2f} MB\n")
    # log_file.write(f"Đường dẫn thư mục runs: {runs_destination_path}\n")

print(f"Đã ghi log tại: {log_path}")