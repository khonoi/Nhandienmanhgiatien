# Nhận diện mệnh giá tiền Việt Nam

Dự án nhỏ sử dụng **Decision Tree** và **OpenCV** để nhận diện mệnh giá tiền Việt Nam (10k, 20k, 50k).

---

## 1. Cài đặt thư viện

Mở terminal (CMD/PowerShell) và chạy:

```bash
pip install opencv-python scikit-learn matplotlib joblib
```

---

## 2. Huấn luyện mô hình

Chạy file:

```bash
python train_model.py
```

- Mô hình sẽ được train và lưu lại thành `money_model.pkl`
- Accuracy trên tập test sẽ được in ra.

---

## 3. Dự đoán ảnh mới

Đặt 1 ảnh test vào thư mục data/test sau đó chạy:

```bash
python predict.py
```
