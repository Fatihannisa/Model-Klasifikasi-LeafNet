import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
import io

# =========================
# ----- LOAD MODEL --------
# =========================
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_leafnet.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

LABELS = [
    "Acalypha siamensis", "Andrographis paniculata", "Cananga odorata", "Capsicum sp", "Catharanthus roseus",
    "Dracaena angustifolia", "Ficus microcarpa", "Flueggea virosa", "Gardenia jasminoides", "Leucaena leucocephala",
    "Moringa oleifera", "Orthosiphon aristatus", "Pandanus amaryllifolius", "Phyllanthus amarus",
    "Physalis angulata", "Rosa sp", "Solanum nigrum", "Syzygium polyanthum", "Vernonia amygdalina", "Ziziphus mauritiana"
]

# =========================
# ----- PREDIKSI ---
# =========================
def predict(image: Image.Image):
    image = image.convert("RGB")
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32) 
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    expected = input_details[0]["shape"]
    if list(img.shape) != list(expected):
        st.error(f"Shape input salah: {img.shape}, harusnya {expected}")
        return None, None

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    pred_idx = np.argmax(output)
    confidence = float(np.max(output))

    return LABELS[pred_idx], confidence


# =========================
# ------ USER INTERFACE -------
# =========================
st.set_page_config(page_title="Sistem Identifikasi Herbal Antidiabetes Berbasis LeafNet", layout="wide")

# ---- HEADER ----
st.markdown("""
    <div style="background:#f3f3f3; padding-left:20px; border-radius:0px; width:100%; display:flex; justify-content:space-between; align-items:center;">
        <h1 style='font-size:30px; font-weight:700;'>DiaHerb</h1>
    </div>
    <hr>
""", unsafe_allow_html=True)

# -------------------------------
# PAGE SELECTOR
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "upload"
    
# =======================
# === HALAMAN UPLOAD ====
# =======================
if st.session_state.page == "upload":
    
    st.markdown("""
        <h2 style="margin:0px; font-size:55px; font-weight:600;">Sistem Identifikasi Daun Herbal Antidiabetes Berbasis Model LeafNet</h2>
        <p style="font-size:18px; margin-bottom:40px; width:95%;">DiaHerb merupakan sebuah sistem berbasis website yang dibangun untuk membantu mengidentifikasi daun herbal antidiabetes yang mirip secara morfologi dengan tanaman lain. Sistem ini dibangun menggunakan teknologi Deep Learning dan Computer Vision.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.6,1.2])

    with col1:
        # ---- CSS Upload Area + Hidden Uploader ----
        st.markdown("""
        <style>
        .fake-upload-box {
            padding: 60px;
            border: 3px dashed #888;
            border-radius: 18px;
            text-align: center;
            background: #f5f5f5;
            cursor: pointer;
            transition: 0.3s;
        }
        .fake-upload-box:hover {
            border-color: #4CAF50;
            background: #eef8ee;
        }
        /* Sembunyikan uploader Streamlit */
        div[data-testid="stFileUploader"] {
            opacity: 0 !important;
            height: 1px !important;
            overflow: hidden !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # ---- Fake Upload Box ----
        st.markdown("""
        <div class="fake-upload-box" onclick="document.querySelector('input[type=file]').click()">
            <h1 style="font-size:50px;">📷</h1>
            <h3>Seret & Lepas Gambar Daun</h3>
            <p>atau klik untuk memilih file (JPG/JPEG/PNG)</p>
        </div>
        """, unsafe_allow_html=True)

        # ---- Hidden uploader asli ----
        uploaded_img = st.file_uploader("Upload Gambar", type=["jpg","jpeg","png"])

        # ---- Preview setelah upload ----
        if uploaded_img:
            st.markdown("### 📌 Preview Gambar")
            img_preview = Image.open(uploaded_img)
            st.image(img_preview, width=300)

        # ==== CSS Tombol Bulat ====
        st.markdown("""
        <style>
        .round-btn button {
            background-color: #4CAF50 !important;
            color: white !important;
            padding: 14px 25px !important;
            border-radius: 40px !important;
            border: none !important;
            font-size: 18px !important;
            font-weight: 600 !important;
            margin-top: 20px;
        }
        .round-btn button:hover {
            background-color: #45a049 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # ---- Tombol Identifikasi bulat (berfungsi penuh) ----
        st.markdown('<div class="round-btn">', unsafe_allow_html=True)
        clicked = st.button("🔍 Identifikasi Daun", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if clicked:
            if uploaded_img:
                st.session_state.image = uploaded_img
                st.session_state.page = "result"
                st.rerun()
            else:
                st.warning("Silakan unggah gambar terlebih dahulu.")

    with col2:
        st.markdown("""
            <div style="padding:20px; background:#f2f2f2; border-radius:12px;">
                <b>Tips pengambilan gambar:</b>
                <ul>
                    <li>Daun berada di tengah frame kamera</li>
                    <li>Pencahayaan cukup & tidak terlalu gelap</li>
                    <li>Latar belakang polos, lebih baik putih</li>
                    <li>Daun harus jelas dan memenuhi area foto</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


# =======================
# === HALAMAN HASIL =====
# =======================
elif st.session_state.page == "result":

    st.markdown("""
        <div style="text-align:center; margin-top:10px;">
            <h2 style="margin:0;">Hasil Identifikasi</h2>
        </div>
    """, unsafe_allow_html=True)

    img = Image.open(st.session_state.image)
    pred_name, conf = predict(img)

    colA, colB = st.columns([1.5,1])

    # ---- KIRI: Gambar & Detail ----
    with colA:
        st.image(img, caption="Gambar yang diunggah", width=350)
        st.markdown(f"""
            <div style="background:#ededed; padding:18px; border-radius:10px; margin-top:15px;">
                <b>Nama Ilmiah</b><br>
                {pred_name}<br><br>
                <b>Nama Umum:</b><br>
                <ul>
                    <li>Contoh 1</li>
                    <li>Contoh 2</li>
                    <li>Contoh 3</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # ---- KANAN: STATUS ----
    with colB:
        st.markdown(f"""
            <div style="background:#ededed; padding:18px; border-radius:10px;">
                <b>Status</b><br>
                Tanaman obat antidiabetes<br><br>

                <b>Tingkat kepercayaan sistem</b><br>
                {conf*100:.2f}%
            </div>
        """, unsafe_allow_html=True)

    # ---- INFORMASI ----
    st.markdown("""
        <div style="background:#f2f2f2; padding:20px; border-radius:10px; margin-top:25px;">
            <b>Informasi (jika herbal antidiabetes)</b><br>
            <p>Tambahkan informasi herbal di sini...</p>
        </div>
    """, unsafe_allow_html=True)

    # ---- LINK ----
    st.text_input("Tautan artikel", "https://contoh-artikel.com")
    st.text_input("Tautan jurnal penelitian", "https://contoh-jurnal.com")

    # ---- Cara mengolah ----
    st.markdown("""
        <div style="background:#f2f2f2; padding:20px; border-radius:10px; margin-top:20px;">
            <b>Cara mengolah herbal antidiabetes</b><br>
            1. langkah 1<br>
            2. langkah 2<br>
            3. langkah 3<br>
            4. langkah x<br>
        </div>
    """, unsafe_allow_html=True)

    st.button("⬅️Kembali", on_click=lambda: (st.session_state.update({"page": "upload"}), st.rerun()))


# ---- FOOTER ----
st.markdown("<hr><center>©2025 | Klasifikasi Herbal Antidiabetes Berbasis Model LeafNet | 211401034 | Listy Zulmi</center>", unsafe_allow_html=True)
