import cv2
import glob
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Danh sách mệnh giá
labels = ["10k", "20k", "50k"]

# Hàm trích xuất histogram màu
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    hist = cv2.calcHist([img], [0, 1, 2], None,
                        [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# -------------------
# 1. Chuẩn bị dữ liệu
# -------------------
X, y = [], []
for label, money in enumerate(labels):
    for img_path in glob.glob(f"data/{money}/*.jpg"):
        features = extract_features(img_path)
        X.append(features)
        y.append(label)

# -------------------
# 2. Tách train/test (70%/30%)
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Tổng dữ liệu: {len(X)} ảnh")
print(f"Train: {len(X_train)} ảnh, Test: {len(X_test)} ảnh")

# -------------------
# 3. Huấn luyện mô hình
# -------------------
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

# -------------------
# 4. Đánh giá trên tập test
# -------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Độ chính xác trên tập test:", acc)

# -------------------
# 5. Lưu mô hình đã train
# -------------------
joblib.dump(model, "money_model.pkl")
print("✅ Mô hình đã lưu vào file money_model.pkl")
