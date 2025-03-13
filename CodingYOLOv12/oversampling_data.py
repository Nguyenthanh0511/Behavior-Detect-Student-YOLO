import os
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch

# 1. Định nghĩa đường dẫn cho 3 tệp train, valid, test
train_path = "/home/minhnv/Documents/ntt/AiIot/FinalPorject/CodingYOLOv12/Dataset/Student-Behavior-Recognition-5/train/images"
valid_path = "/home/minhnv/Documents/ntt/AiIot/FinalPorject/CodingYOLOv12/Dataset/Student-Behavior-Recognition-5/valid/images"
test_path  = "/home/minhnv/Documents/ntt/AiIot/FinalPorject/CodingYOLOv12/Dataset/Student-Behavior-Recognition-5/test/images"

labels_train_path = "/home/minhnv/Documents/ntt/AiIot/FinalPorject/CodingYOLOv12/Dataset/Student-Behavior-Recognition-5/train/labels"
labels_valid_path = "/home/minhnv/Documents/ntt/AiIot/FinalPorject/CodingYOLOv12/Dataset/Student-Behavior-Recognition-5/valid/labels"
labels_test_path  = "/home/minhnv/Documents/ntt/AiIot/FinalPorject/CodingYOLOv12/Dataset/Student-Behavior-Recognition-5/test/labels"

# 2. Định nghĩa các pipeline augmentation cho train, valid và test
transform_train = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(
        translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
        scale=(0.9, 1.1),
        rotate=(-15, 15),
        p=0.5
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.0))

transform_valid = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

# 3. Xây dựng Custom Dataset cho dữ liệu YOLO với thư mục nhãn riêng
class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        """
        images_dir: Thư mục chứa ảnh.
        labels_dir: Thư mục chứa file nhãn (mỗi ảnh có file .txt cùng tên).
        transform: Các phép biến đổi augmentation áp dụng.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Đọc file nhãn từ thư mục labels_dir
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_file)
        bboxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        bbox = list(map(float, parts[1:5]))
                        bboxes.append(bbox)
                        labels.append(cls)
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        
        target = {'bboxes': bboxes, 'labels': labels}
        return image, target

# 4. Tạo dataset cho các tập train, valid và test
train_dataset = YOLODataset(images_dir=train_path, labels_dir=labels_train_path, transform=transform_train)
valid_dataset = YOLODataset(images_dir=valid_path, labels_dir=labels_valid_path, transform=transform_valid)
test_dataset  = YOLODataset(images_dir=test_path, labels_dir=labels_test_path, transform=transform_valid)

# 5. Áp dụng oversampling cho tập train (ví dụ áp dụng dựa trên tần suất xuất hiện của lớp)
class_counts = {}
for idx in range(len(train_dataset)):
    _, target = train_dataset[idx]
    for label in target['labels']:
        class_counts[label] = class_counts.get(label, 0) + 1
print("Số lượng instance cho các lớp:", class_counts)

# Thiết lập tiêu chí: ví dụ, nếu một lớp có số mẫu nhỏ hơn 50% số mẫu trung bình, coi là underrepresented.
if len(class_counts) > 0:
    avg_count = np.mean(list(class_counts.values()))
    threshold = 0.5 * avg_count
else:
    threshold = 0
underrepresented_labels = [label for label, count in class_counts.items() if count < threshold]
print("Các lớp underrepresented:", underrepresented_labels)

sample_weights = []
for idx in range(len(train_dataset)):
    _, target = train_dataset[idx]
    if len(target['labels']) == 0:
        weight = 0.1  # Ảnh không có nhãn nhận trọng số thấp
    else:
        # Nếu ảnh chứa ít nhất một lớp underrepresented, tăng trọng số dựa trên nghịch đảo số mẫu của lớp đó.
        underrep_in_image = [label for label in target['labels'] if label in underrepresented_labels]
        if len(underrep_in_image) > 0:
            weights = [1.0 / class_counts[label] for label in underrep_in_image]
            weight = max(weights)
        else:
            weight = 0.1  # Nếu không có lớp underrepresented, gán trọng số thấp
    sample_weights.append(weight)

sample_weights = np.array(sample_weights)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# 6. Tạo DataLoader cho train, valid và test
train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, collate_fn=lambda x: tuple(zip(*x)))
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# 7. Kiểm tra duyệt qua một batch từ mỗi loader
print("Train batch:")
for images, targets in train_loader:
    print("Số lượng ảnh trong batch:", len(images))
    break

print("Validation batch:")
for images, targets in valid_loader:
    print("Số lượng ảnh trong batch:", len(images))
    break

print("Test batch:")
for images, targets in test_loader:
    print("Số lượng ảnh trong batch:", len(images))
    break

# 8. Export các ảnh đã được tăng cường từ tập train
# Lưu ý: Vì transform_train có ToTensorV2 và Normalize, ta cần đảo ngược normalization để hiển thị ảnh.
export_folder = "/home/minhnv/Documents/ntt/AiIot/FinalPorject/CodingYOLOv12/Dataset/Student-Behavior-Recognition-5/augmented_images_train"
os.makedirs(export_folder, exist_ok=True)

# Hàm đảo ngược normalization để chuyển về dạng ảnh uint8
def denormalize_image(image_tensor):
    # image_tensor: torch.Tensor có dạng [C, H, W]
    means = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    stds = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image_tensor = image_tensor * stds + means  # đảo normalization
    image_tensor = image_tensor * 255.0
    image_tensor = image_tensor.clamp(0, 255).byte()
    image_np = image_tensor.permute(1,2,0).cpu().numpy()  # chuyển từ [C, H, W] sang [H, W, C]
    # cv2 lưu ảnh với định dạng BGR nên chuyển đổi từ RGB sang BGR
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_bgr

# Xuất một số ảnh tăng cường (ở đây export tất cả các ảnh trong train_dataset)
for i in range(len(train_dataset)):
    image, _ = train_dataset[i]  # ảnh đã được augmentation, ở dạng tensor nhờ ToTensorV2
    image_bgr = denormalize_image(image)
    output_path = os.path.join(export_folder, f"augmented_{i}.jpg")
    cv2.imwrite(output_path, image_bgr)
