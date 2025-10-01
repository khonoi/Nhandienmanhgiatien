import cv2
import numpy as np
import joblib
import glob
import os

labels = ["10k", "20k", "50k"]

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    img = cv2.resize(img, (128, 128))
    hist = cv2.calcHist([img], [0, 1, 2], None,
                        [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Load mô hình đã lưu
model = joblib.load("money_model.pkl")

# Thư mục test
test_folder = "data/test/"

# Lấy tất cả file ảnh trong test/
image_paths = glob.glob(os.path.join(test_folder, "*.*"))

if not image_paths:
    print("⚠️ Không tìm thấy ảnh nào trong thư mục data/test/")
else:
    for img_path in image_paths:
        try:
            features = extract_features(img_path)
            pred = model.predict([features])[0]
            print(f"{os.path.basename(img_path)} → {labels[pred]}")
        except Exception as e:
            print(f"Lỗi với ảnh {img_path}: {e}")
