# Sử dụng Python 3.9
# tại sao dùng python:3.9-slim?
FROM python:3.9-slim

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    gfortran \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục app và copy mã nguồn vào
WORKDIR /app
COPY . /app

# Cài đặt các thư viện Python
RUN pip install --upgrade pip
ENV PIP_ONLY_BINARY=":all:"
RUN pip install --no-cache-dir --only-binary=:all: "numpy<2"
RUN pip install --no-cache-dir "pandas==1.2.2"
RUN pip install --no-cache-dir -r requirements.txt
# Cài đặt streamlit rõ ràng để tránh lỗi PATH
RUN pip install --no-cache-dir streamlit

# Mở port mặc định của Streamlit
EXPOSE 8501

# Chạy ứng dụng Streamlit
CMD ["streamlit", "run", "eval_app.py", "--server.port=8501", "--server.address=0.0.0.0"]