import streamlit as st
import os
import gdown
import zipfile
from PIL import Image
import pandas as pd

MODEL_ZIP_URL = "https://drive.google.com/uc?id=1EAYJ4MjnpBfbvRZrb2pubYrRPQpmu-mq"
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
    "CARNetvHCAttention": {"decoder": Decoder.RNetvHCAttention, "attention": "SoftAttention", "attention_dim": 1024, "encoder_dim": 2048, "hidden_dim": 1024}
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

# ==================== MAIN UI ====================
if mode == "Ảnh đơn":
    st.subheader("📥 Tải ảnh lên để sinh caption")
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đã chọn", use_column_width=True)
        if st.button("📌 Sinh caption"):
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
    st.subheader("📂 Đánh giá nhiều ảnh trong thư mục")
    dataset_folder = st.text_input("📁 Nhập đường dẫn thư mục chứa ảnh", "./dataset")

    if st.button("🚀 Bắt đầu đánh giá"):
        image_dir = os.path.join(dataset_folder, IMAGES_SUBDIRECTORY_NAME)
        if not os.path.exists(image_dir):
            st.error("❌ Không tìm thấy thư mục ảnh.")
        else:
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            results = []

            for file in image_files:
                try:
                    img = Image.open(os.path.join(image_dir, file)).convert("RGB")
                    tokens = net.eval_image_caption(img, vocab)
                    caption = " ".join([w for w in tokens if w not in ("<START>", "<END>")])
                    results.append({"image_name": file, "caption": caption})
                except Exception as e:
                    results.append({"image_name": file, "caption": f"ERROR: {e}"})

            df = pd.DataFrame(results)
            st.success("🎉 Đã xử lý xong toàn bộ ảnh!")
            st.dataframe(df)
            df.to_csv("eval_results.csv", sep="|", index=False)
            st.info("📥 Kết quả đã lưu tại file eval_results.csv")

# ==================== WATERMARK ====================
st.markdown("<div style='text-align:right; color: #888; font-size: 14px;'>Made by <b>Thisorp</b></div>", unsafe_allow_html=True)
