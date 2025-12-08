import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

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
    img = image.resize((224, 224))     # sesuaikan ukuran input model
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
st.set_page_config(page_title="Identifikasi Herbal", layout="wide")

# ---- HEADER ----
st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <h1 style='font-size:32px; font-weight:700;'>DiaHerb</h1>
        <div style="padding:10px 25px; background:#ddd; border-radius:6px; font-weight:600;">
            Navbar
        </div>
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
        <div style="background:#f3f3f3; padding:18px; border-radius:8px; width:90%;">
            <h2 style="margin:0;">Deskripsi</h2>
        </div>
        <p style="margin-top:10px;">Perintah unggah</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.8,1])

    with col1:
        st.markdown("""
            <div style="padding:40px; background:#e0e0e0; border-radius:15px; text-align:center;">
                <h1 style="font-size:55px; margin:0;">📷</h1>
                <p>Unggah gambar daun (JPG/PNG)</p>
            </div>
        """, unsafe_allow_html=True)

        uploaded_img = st.file_uploader("", type=["jpg", "jpeg", "png"])

        st.markdown("""
            <div style="text-align:center; margin-top:20px;">
                <button style="
                    padding:10px 30px;
                    background:#dcdcdc;
                    border:none;
                    border-radius:40px;
                    font-size:18px;
                    cursor:pointer;">
                    Identifikasi
                </button>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Identifikasi", use_container_width=True):
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
                    <li>tips 1</li>
                    <li>tips 2</li>
                    <li>tips 3</li>
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

    st.button("Kembali", on_click=lambda: (st.session_state.update({"page": "upload"}), st.rerun()))


# ---- FOOTER ----
st.markdown("<hr><center>@2025 | KLASIFIKASI HERBAL ANTIDIABETES BERBASIS MODEL LEAFNET | LISTY ZULMI</center>", unsafe_allow_html=True)
