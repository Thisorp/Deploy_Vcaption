# UNETI VCaption - Sinh chú thích ảnh tiếng Việt

## 🚀 Demo ứng dụng - nhánh này build local trên MácOS
## Link Source: https://github.com/Thisorp/Uneti_Vcaption
Bạn có thể chạy demo ứng dụng sinh caption ảnh tiếng Việt bằng Streamlit:
- **Chạy trên máy tính:**
    1. Cài Python 3.9, clone repo này về máy.
    2. Cài thư viện: `pip install -r requirements.txt`
    3. Chạy: `streamlit run eval_app.py`
- **Chạy online:**
    - Deploy lên [Streamlit Cloud](https://streamlit.io/cloud) (kết nối repo GitHub, app sẽ tự động tải model từ Google Drive).

## 📦 Cách sử dụng ứng dụng eval_app.py

### 1. Sinh caption cho ảnh đơn
- Chọn chế độ "Ảnh đơn" trên sidebar.
- Upload ảnh (jpg, jpeg, png).
- Nhấn "📌 Sinh caption" để nhận kết quả chú thích ảnh.

### 2. Đánh giá nhiều ảnh bằng cách upload thư mục
- Chọn chế độ "Toàn bộ thư mục" trên sidebar.
- Upload file .zip chứa các ảnh cần đánh giá (không cần nén thư mục con, chỉ cần các file ảnh trong .zip).
- Nhấn "Bắt đầu đánh giá" để nhận kết quả caption cho tất cả ảnh (hiển thị trực tiếp trên giao diện, không lưu file csv).

### 3. Lưu ý về model
- Model sẽ tự động tải về từ Google Drive khi chạy lần đầu (không cần upload thủ công).
- Không cần đẩy file model lên GitHub.

## ⚙️ Các chế độ và tuỳ chọn
- Chọn mô hình: CaRNetvI, CaRNetvH, CaRNetvHC, CaRNetvHCAttention.
- Chọn giao diện Light/Dark.

## 🛠️ Cài đặt và triển khai

1. **Clone repo:**
    ```bash
    git clone https://github.com/Thisorp/Uneti_Vcaption.git
    cd Uneti_Vcaption
    ```
2. **Cài thư viện:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Chạy ứng dụng:**
    ```bash
    streamlit run eval_app.py
    ```

## 💡 Lưu ý
- Ứng dụng chỉ hỗ trợ upload file .zip cho chế độ đánh giá nhiều ảnh.
- Kết quả sẽ hiển thị trực tiếp trên giao diện, không lưu file csv.
- Nếu deploy trên Streamlit Cloud, không cần chỉnh sửa gì thêm.

## 🌐 Source code
- Tác giả: Thisorp
- Dự án: https://github.com/Thisorp/Uneti_Vcaption
