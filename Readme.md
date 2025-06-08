# VN Captioning
# Hướng dẫn sử dụng mô hình CaRNet
## Link DATASET và TESTSET: 
```bash
https://drive.google.com/file/d/1Y9nuHtmO0p0Jd2lvzIF5euiemkJHJ2IU/view?usp=drive_link
```
## 🚀 Huấn luyện mô hình (Training)

### CaRNetvI
```bash
python main.py RNetvI train 0 1024 --dataset_folder ./dataset --device cuda:0 --epochs 150
```

### CaRNetvH
```bash
python main.py RNetvH train 1024 1024 --dataset_folder ./dataset --device cuda:0 --epochs 150
```

### CaRNetvHC
```bash
python main.py RNetvHC train 1024 1024 --dataset_folder ./dataset --device cuda:0 --epochs 150
```

### CaRNetvHCAttention
```bash
python main.py RNetvHCAttention train 1024 1024 --dataset_folder ./dataset --device cuda:0 --epochs 150 --attention t --attention_dim 1024
```

---

## 🔍 Đánh giá mô hình (Single Image Evaluation)

### CaRNetvI
```bash
python eval.py RNetvI eval 5078 1024 --image_path ./33465647.jpg
```

### CaRNetvH
```bash
python eval.py RNetvH eval 1024 1024 --image_path ./33465647.jpg
```

### CaRNetvHC
```bash
python eval.py RNetvHC eval 1024 1024 --image_path ./33465647.jpg
```

### CaRNetvHCAttention
```bash
python eval.py RNetvHCAttention eval 1024 1024 --attention t --attention_dim 1024 --image_path ./33465647.jpg
```

---

## 📁 Đánh giá toàn bộ thư mục ảnh (Folder Evaluation)

```bash
python eval.py RNetvHCAttention eval 1024 1024 --attention t --attention_dim 1024 --dataset_folder ./testset --output_csv ./testset/caption_test4.csv
```

> 📄 File `results.csv` sẽ được tạo trong đường dẫn `--output_csv`, theo định dạng: `image_name| comment_number| comment`.

---

## 📊 Đánh giá độ chính xác BLEU

```bash
python bleu_newest.py ./caption_test.csv ./results.csv
```

> So sánh giữa nhãn thực tế (`caption_test.csv`) và dự đoán từ mô hình (`results.csv`).

---

📁 **Chú ý cấu trúc thư mục:**
- `./dataset/` hoặc `./testset/`
  - `images/` (thư mục chứa ảnh)
  - `results.csv` (file chú thích, phân tách bằng dấu `|`)

---

💡 **Lưu ý thêm:**
- `--attention t` dùng để bật chế độ attention cho mô hình `CaRNetvHCAttention`.
- `--attention_dim` nên để 1024 (mặc định khuyến nghị).
- Nếu không cung cấp `--image_path`, chương trình sẽ duyệt toàn bộ thư mục `images/`.