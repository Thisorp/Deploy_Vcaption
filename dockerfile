# Sử dụng Python 3.9
FROM python:3.9-slim

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục app và copy mã nguồn vào
WORKDIR /app
COPY . /app

# Cài đặt các thư viện Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Mở port mặc định của Streamlit
EXPOSE 8501

# Chạy ứng dụng Streamlit
CMD ["streamlit", "run", "eval_app.py", "--server.port=8501", "--server.address=0.0.0.0"]