import streamlit as st
import os
import gdown
import zipfile
from PIL import Image
import pandas as pd
import threading
from prometheus_client import start_http_server, Counter, Histogram

MODEL_ZIP_URL = "https://drive.google.com/uc?id=1wlOyOf--30DgjVQzWJR2Nv811sF7zvVe"
ZIP_PATH = "./model_saved.zip"
MODEL_DIR = "./.saved"

def ensure_model():
    if not os.path.exists(MODEL_DIR):
        st.warning("Đang tải model từ Google Drive, vui lòng chờ...")
        gdown.download(MODEL_ZIP_URL, ZIP_PATH, quiet=False)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(ZIP_PATH)
        st.success("Tải và giải nén model thành công!")

ensure_model()

# Import các thư viện liên quan torch SAU khi model đã sẵn sàng
import torch
from NeuralModels.FactoryModels import *
from NeuralModels.Vocabulary import Vocabulary
from VARIABLE import IMAGES_SUBDIRECTORY_NAME
from NeuralModels.Attention.SoftAttention import SoftAttention

# ==================== SIDEBAR OPTIONS ====================
st.sidebar.header("⚙️ Tuỳ chọn")
model_key = st.sidebar.selectbox("Chọn mô hình:", list({
    "CaRNetvI", "CaRNetvH", "CaRNetvHC", "CaRNetvHCAttention"
}))
mode = st.sidebar.radio("Chế độ đánh giá:", ["Ảnh đơn", "Toàn bộ thư mục"])
theme_mode = st.sidebar.radio("🎨 Giao diện", ["Light", "Dark"])

# ==================== THEME MODE ====================
if theme_mode == "Dark":
    st.markdown("""<style>
        body, .stApp { background-color: #0e1117; color: #cfd6e1; }
        h1, h2, h3, .css-1v0mbdj, .css-1d391kg { color: #4fc3f7; }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
        body, .stApp { background-color: #ffffff; color: #000000; }
        h1, h2, h3 { color: #0056b3; }
    </style>""", unsafe_allow_html=True)

# ==================== TITLE & LOGO ====================
logo = Image.open("Uneti-2.png")
st.image(logo, width=90)
st.markdown("""
    <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px;'>
        <h1 style='padding-top: 10px; color: #0056b3;'>UNETI VCaption 📷</h1>
        <p style='font-size:18px; color: #333;'>Hệ thống sinh chú thích ảnh tiếng Việt sử dụng mô hình CaRNet</p>
    </div>
""", unsafe_allow_html=True)


# ==================== MODEL CONFIG ====================
MODEL_CONFIGS = {
    "CaRNetvI": {"decoder": Decoder.RNetvI, "attention": None, "attention_dim": 0, "encoder_dim": 2086, "hidden_dim": 1024},
    "CaRNetvH": {"decoder": Decoder.RNetvH, "attention": None, "attention_dim": 0, "encoder_dim": 1024, "hidden_dim": 1024},
    "CaRNetvHC": {"decoder": Decoder.RNetvHC, "attention": None, "attention_dim": 0, "encoder_dim": 1024, "hidden_dim": 1024},
    "CaRNetvHCAttention": {"decoder": Decoder.RNetvHCAttention, "attention": "SoftAttention", "attention_dim": 1024, "encoder_dim": 2048, "hidden_dim": 1024}
}

# Thay thế st.cache bằng st.cache_resource (theo khuyến nghị mới của Streamlit)
@st.cache_resource
def load_model(model_key):
    cfg = MODEL_CONFIGS[model_key]
    encoder = FactoryEncoder(Encoder.CResNet50Attention if cfg["attention"] else Encoder.CResNet50)
    decoder = FactoryDecoder(cfg["decoder"])
    attention = SoftAttention if cfg["attention"] == "SoftAttention" else None
    vocab = Vocabulary()
    net = FactoryNeuralNet(NeuralNet.CaRNet)(
        encoder=encoder,
        decoder=decoder,
        attention=attention,
        attention_dim=cfg["attention_dim"],
        net_name=model_key,
        encoder_dim=cfg["encoder_dim"],
        hidden_dim=cfg["hidden_dim"],
        padding_index=vocab.predefined_token_idx()["<PAD>"],
        vocab_size=len(vocab.word2id.keys()),
        embedding_dim=vocab.embeddings.shape[1],
        device="cpu"
    )
    net.load("./.saved")
    return net, vocab

net, vocab = load_model(model_key)
st.markdown("---")

# ==================== METRICS ====================
@st.cache_resource
def get_metrics():
    REQUEST_COUNT = Counter('caption_requests_total', 'Tổng số request caption')
    CAPTION_LATENCY = Histogram('caption_latency_seconds', 'Thời gian xử lý caption')
    return REQUEST_COUNT, CAPTION_LATENCY

REQUEST_COUNT, CAPTION_LATENCY = get_metrics()

def start_metrics_server():
    # Prometheus sẽ scrape metrics tại http://localhost:8000/metrics
    start_http_server(8000)

# Chạy metrics server ở background
threading.Thread(target=start_metrics_server, daemon=True).start()

# ==================== MAIN UI ====================
if mode == "Ảnh đơn":
    st.subheader("📥 Tải ảnh lên để sinh caption")
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đã chọn", use_column_width=True)
        if st.button("📌 Sinh caption"):
            with CAPTION_LATENCY.time():
                REQUEST_COUNT.inc()
                with st.spinner("⏳ Đang xử lý..."):
                    tokens = net.eval_image_caption(image, vocab)
                    caption = " ".join([w for w in tokens if w not in ("<START>", "<END>")])

                    st.markdown("""
                    <div style='background-color: #e8f5e9; padding: 20px; border-radius: 8px; margin-top: 20px;'>
                        <h4 style='color: green; margin-bottom: 10px;'>✅ Caption:</h4>
                        <p style='font-size: 20px; font-weight: bold; color: #1b5e20;'>""" + caption + """</p>
                    </div>
                    """, unsafe_allow_html=True)

elif mode == "Toàn bộ thư mục":
    st.subheader("📂 Đánh giá nhiều ảnh bằng cách upload thư mục")
    uploaded_folder = st.file_uploader("Chọn folder ảnh (nén .zip)", type=["zip"])
    temp_extract_dir = "./uploaded_images"
    if uploaded_folder is not None:
        # Xóa thư mục tạm nếu đã tồn tại
        if os.path.exists(temp_extract_dir):
            import shutil
            shutil.rmtree(temp_extract_dir)
        os.makedirs(temp_extract_dir, exist_ok=True)
        # Lưu file zip và giải nén
        with open("uploaded_images.zip", "wb") as f:
            f.write(uploaded_folder.read())
        with zipfile.ZipFile("uploaded_images.zip", 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        os.remove("uploaded_images.zip")
        st.success("Đã upload và giải nén thư mục ảnh!")
        image_dir = temp_extract_dir
        # Duyệt đệ quy lấy tất cả file ảnh trong mọi thư mục con
        image_files = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('jpg', 'jpeg', 'png')):
                    image_files.append(os.path.join(root, f))
        if not image_files:
            st.error("Không tìm thấy ảnh hợp lệ (.jpg, .jpeg, .png) trong file .zip!")
        elif st.button("🚀 Bắt đầu đánh giá"):
            with CAPTION_LATENCY.time():
                REQUEST_COUNT.inc(len(image_files))
                results = []
                for file_path in image_files:
                    file_name = os.path.relpath(file_path, image_dir)
                    try:
                        img = Image.open(file_path).convert("RGB")
                        tokens = net.eval_image_caption(img, vocab)
                        caption = " ".join([w for w in tokens if w not in ("<START>", "<END>")])
                        results.append({"image_name": file_name, "caption": caption})
                    except Exception as e:
                        results.append({"image_name": file_name, "caption": f"ERROR: {e}"})
                df = pd.DataFrame(results)
                st.success("🎉 Đã xử lý xong toàn bộ ảnh!")
                st.dataframe(df)
    else:
        st.info("Vui lòng upload file .zip chứa các ảnh để đánh giá.")

# ==================== WATERMARK ====================
st.markdown("<div style='text-align:right; color: #888; font-size: 14px;'>Made by <b>Thisorp</b></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:right; color: #888; font-size: 14px;'>Source dự án <b>https://github.com/Thisorp/Uneti_Vcaption</b></div>", unsafe_allow_html=True)
