import os
import cv2
import numpy as np

IMG_SIZE = 128
TRAIN_DIR = "Data/chest_xray/train"
VAL_DIR = "Data/chest_xray/val"
TEST_DIR = "Data/chest_xray/test"


def load_data(data_dir, img_size=IMG_SIZE):
    x_data = []
    y_data = []

    # Duyệt qua các nhãn: 0 cho NORMAL, 1 cho PNEUMONIA
    classes = ["NORMAL", "PNEUMONIA"]
    for label, class_name in enumerate(classes):
        path = os.path.join(data_dir, class_name)
        if not os.path.exists(path):
            continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            # Tiền xử lý
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0  # Normalize

            x_data.append(img.flatten())
            y_data.append(label)

    return np.array(x_data), np.array(y_data)
