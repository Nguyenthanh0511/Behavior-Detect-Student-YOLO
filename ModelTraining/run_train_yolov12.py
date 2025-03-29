#------------------ Hyperparameter Tuning
import itertools
import json
from datetime import datetime
import torch
import os
import yaml
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from ultralytics import YOLO
import numpy as np
from collections import Counter
import logging
import sys
import time
from pathlib import Path

# Thiết lập logging với đường dẫn đầy đủ
def setup_logging(experiment_path):
    log_file = os.path.join(experiment_path, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Log file saved to: {log_file}")

# Thiết lập random seed cho reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Thiết lập seed
set_seed()

# Đường dẫn đến dataset
data_yaml_path = "/hdd2/minhnv/CodingYOLOv12/Dataset/T-Student_FIT-DNU-1/data.yaml"

# Define hyperparameter configurations to try
hyperparameter_configs = [
    # Configuration dictionaries - each represents a complete experiment
    {
        "name": "baseline",
        "model_size": "yolov12s",  # Try different model sizes: yolov8n, yolov8s, yolov8m, yolov8l
        "epochs": 300,
        "patience": 30,
        "imgsz": 640,
        "batch": 32,
        "optimizer": "Adam",  # Try: 'Adam', 'SGD', 'AdamW'
        "lr0": 0.01,
        "lrf": 0.1,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "augmentation": "default",  # Controls which augmentation settings to use
    },
    {
        "name": "high_lr",
        "model_size": "yolov12s",
        "epochs": 300,
        "patience": 30,
        "imgsz": 640,
        "batch": 32,
        "optimizer": "Adam",
        "lr0": 0.03,  # Higher learning rate
        "lrf": 0.1,
        "weight_decay": 0.0005,
        "warmup_epochs": 5.0,  # Longer warmup for higher LR
        "augmentation": "default",
    },
    {
        "name": "sgd_optimizer",
        "model_size": "yolov12s",
        "epochs": 300,
        "patience": 30,
        "imgsz": 640,
        "batch": 32,
        "optimizer": "SGD",  # Changed optimizer
        "lr0": 0.01,
        "lrf": 0.01,  # Slower LR decay
        "momentum": 0.95,  # Higher momentum for SGD
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "augmentation": "default",
    },
    {
        "name": "heavy_aug",
        "model_size": "yolov12s",
        "epochs": 300,
        "patience": 30,
        "imgsz": 640,
        "batch": 32,
        "optimizer": "Adam",
        "lr0": 0.01,
        "lrf": 0.1,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "augmentation": "heavy",  # More aggressive augmentation
    },
    {
        "name": "larger_model",
        "model_size": "yolov12s",  # Larger model architecture
        "epochs": 300,
        "patience": 30,
        "imgsz": 640,
        "batch": 16,  # Reduced batch size for larger model
        "optimizer": "Adam",
        "lr0": 0.01,
        "lrf": 0.1,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "augmentation": "default",
    },
    {
        "name": "longer_training",
        "model_size": "yolov12s",
        "epochs": 500,  # More epochs
        "patience": 50,  # Higher patience
        "imgsz": 640,
        "batch": 32,
        "optimizer": "Adam",
        "lr0": 0.01,
        "lrf": 0.01,  # Slower LR decay for longer training
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "augmentation": "default",
    },
]

# Augmentation configurations
augmentation_configs = {
    "default": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0001,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
        "erasing": 0.4,
    },
    "light": {
        "hsv_h": 0.01,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 5.0,
        "translate": 0.05,
        "scale": 0.2,
        "shear": 1.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.7,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.2,
    },
    "heavy": {
        "hsv_h": 0.02,
        "hsv_s": 0.9,
        "hsv_v": 0.6,
        "degrees": 15.0,
        "translate": 0.2,
        "scale": 0.7,
        "shear": 3.0,
        "perspective": 0.001,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.3,
        "copy_paste": 0.3,
        "erasing": 0.6,
    }
}

# Results tracking
experiment_results = []

# Run each hyperparameter configuration
for config_idx, config in enumerate(hyperparameter_configs):
    print(f"\n\n{'='*50}")
    print(f"Running experiment {config_idx+1}/{len(hyperparameter_configs)}: {config['name']}")
    print(f"{'='*50}")
    
    # Set model size
    nameYoloFamily = config["model_size"]
    
    # Create unique save path for this experiment
    experiment_date = datetime.now().strftime("%d%m%Y_%H%M%S")
    experiment_save_path = f'/hdd2/minhnv/CodingYOLOv12/Behavior-Detect-Student-YOLO/StaticModels/{nameYoloFamily}_{experiment_date}_exp{config_idx+1}_{config["name"]}/'
    os.makedirs(experiment_save_path, exist_ok=True)
    
    # Setup logging cho experiment này
    setup_logging(experiment_save_path)
    
    # Save configuration for reference
    with open(os.path.join(experiment_save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Initialize model
    model = YOLO(f'{nameYoloFamily}.yaml')
    
    # Change working directory
    os.chdir(experiment_save_path)
    
    # Get augmentation settings
    aug_config = augmentation_configs[config.get("augmentation", "default")]
    
    # Prepare training arguments
    train_args = {
        "data": data_yaml_path,
        "epochs": config["epochs"],
        "patience": config["patience"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "optimizer": config["optimizer"],
        "lr0": config["lr0"],
        "lrf": config["lrf"],
        "weight_decay": config["weight_decay"],
        "warmup_epochs": config["warmup_epochs"],
        "pretrained": True,
        "dropout": 0.0,
        "save_period": -1,  # Không lưu các epoch trung gian
        "save_best": True,  # Lưu model tốt nhất
        "save_last": False  # Không lưu model cuối cùng
        # Add all augmentation parameters
        **aug_config
    }
    
    # Add momentum if using SGD
    if config["optimizer"] == "SGD" and "momentum" in config:
        train_args["momentum"] = config["momentum"]
    
    # Add learning rate scheduler parameters
    train_args.update({
        "cos_lr": True,  # Cosine learning rate schedule
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1
    })
    
    # Add model checkpointing parameters
    train_args.update({
        "save_period": 2,  # Save every 50 epochs
        "box": 7.5,    # Box loss weight
        "cls": 0.5,    # Classification loss weight
        "dfl": 1.5     # DFL loss weight
    })
    
    # Add NMS parameters
    train_args.update({
        "conf": 0.25,  # Confidence threshold
        "iou": 0.7,    # NMS IoU threshold
        "max_det": 300 # Maximum number of detections
    })
    
    # Train model with this configuration
    print(f"Training with configuration: {config['name']}")
    print(json.dumps(train_args, indent=2))
    
    try:
        results = model.train(**train_args)
    except Exception as e:
        logging.error(f"Error in experiment {config['name']}: {str(e)}")
        logging.error(f"Configuration: {json.dumps(config, indent=2)}")
        continue
    
    # Save model
    model_filename = f'{nameYoloFamily}_{config["name"]}.pt'
    full_model_path = os.path.join(experiment_save_path, model_filename)
    model.save(full_model_path)
    
    # Calculate model size
    model_size_mb = os.path.getsize(full_model_path) / (1024 * 1024)
    
    # Store results
    experiment_result = {
        "experiment_name": config["name"],
        "model_path": full_model_path,
        "model_size_mb": model_size_mb,
        "metrics": results,
        "early_stopping_epoch": results.epoch,
        "best_epoch": results.best_epoch,
        "training_time": results.duration,
    }
    experiment_results.append(experiment_result)
    
    # Log results
    log_path = os.path.join(experiment_save_path, f'training_log_{model_filename}.txt')
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Experiment: {config['name']}\n")
        log_file.write(f"Model: {nameYoloFamily}\n")
        log_file.write(f"Date: {experiment_date}\n")
        log_file.write(f"Model size: {model_size_mb:.2f} MB\n")
        log_file.write(f"Configuration:\n{json.dumps(config, indent=2)}\n")
        log_file.write(f"Training arguments:\n{json.dumps(train_args, indent=2)}\n")
        # Add more metrics as needed
        log_file.write(f"Validation mAP50: {results.val.map50}\n")
        log_file.write(f"Validation mAP50-95: {results.val.map}\n")
        log_file.write(f"Validation Precision: {results.val.precision}\n")
        log_file.write(f"Validation Recall: {results.val.recall}\n")
        log_file.write(f"GPU Memory Used: {get_gpu_memory():.2f} MB\n")

# Save all experiment results to compare them
final_results_path = os.path.join('/hdd2/minhnv/CodingYOLOv12/Behavior-Detect-Student-YOLO/StaticModels/', f'all_experiments_{datetime.now().strftime("%d%m%Y")}.json')
with open(final_results_path, 'w') as f:
    json.dump(experiment_results, f, indent=4)

print(f"\nAll experiments completed. Results saved to {final_results_path}")

# Thêm kiểm tra trước khi tìm best experiment
if experiment_results:
    best_experiment = min(experiment_results, key=lambda x: x.get("metrics", {}).get("val_loss", float('inf')))
    print(f"\nBest experiment: {best_experiment['experiment_name']}")
    print(f"Best model path: {best_experiment['model_path']}")
else:
    print("\nNo experiments completed successfully!")

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0

def run_cross_validation(config, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        # Train model for this fold
        results = model.train(**train_args)
        cv_results.append(results)
    
    return cv_results

def check_gpu():
    if not torch.cuda.is_available():
        logging.warning("GPU not available. Training will be slower!")
        return False
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    return True

# Thêm vào đầu code
if not check_gpu():
    logging.warning("Consider using GPU for faster training!")

def validate_config(config):
    required_fields = ["name", "model_size", "epochs", "batch", "optimizer"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    if config["optimizer"] == "SGD" and "momentum" not in config:
        raise ValueError("SGD optimizer requires momentum parameter")