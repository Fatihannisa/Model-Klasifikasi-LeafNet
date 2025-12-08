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

# Label contoh (sesuaikan)
LABELS = [
    "Andrographis paniculata",  # sambiloto
    "Syzygium polyanthum",      # salam
    "Orthosiphon aristatus"     # kumis kucing
]

# =========================
# ----- FUNGSI PREDIKSI ---
# =========================
def predict(image: Image.Image):
    img = image.resize((224, 224))     # sesuaikan ukuran input model
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    pred_idx = np.argmax(output)
    confidence = float(np.max(output))

    return LABELS[pred_idx], confidence


# =========================
# ------ HALAMAN UI -------
# =========================

st.set_page_config(page_title="Identifikasi Herbal", layout="wide")

# ---- HEADER ----
st.markdown("""
    <h1 style='font-size:32px; font-weight:700;'>Logo + Header</h1>
    <hr>
""", unsafe_allow_html=True)

# -------------------------------
# PAGE SELECTOR (simulate pages)
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "upload"

if st.session_state.page == "upload":
    # =======================
    # === HALAMAN UPLOAD ====
    # =======================

    st.markdown("""
        <div style="background:#f4f4f4; padding:15px; border-radius:6px; width:70%;">
            <h2>Deskripsi</h2>
        </div>
        <p>Perintah unggah</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("""
            <div style="padding:30px; background:#e5e5e5; border-radius:12px; text-align:center;">
                <h3>📷</h3>
                <p>Unggah gambar daun (JPG/PNG)</p>
            </div>
        """, unsafe_allow_html=True)

        uploaded_img = st.file_uploader("", type=["jpg", "jpeg", "png"])

        st.write(input_details)
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
        <div style="text-align:center;">
            <h2>Hasil Identifikasi</h2>
        </div>
    """, unsafe_allow_html=True)

    img = Image.open(st.session_state.image)
    pred_name, conf = predict(img)

    colA, colB = st.columns([2,2])

    # ---- GAMBAR + Nama ----
    with colA:
        st.image(img, caption="Gambar yang diunggah", width=300)
        st.markdown(f"""
            <div style="background:#efefef; padding:15px; border-radius:8px;">
                <b>Nama Ilmiah</b><br>
                {pred_name}<br><br>
                <b>Nama Umum:</b><br>
                1. contoh 1<br>
                2. contoh 2<br>
                3. contoh 3<br>
            </div>
        """, unsafe_allow_html=True)

    # ---- STATUS ----
    with colB:
        st.markdown(f"""
            <div style="background:#efefef; padding:15px; border-radius:8px;">
                <b>Status</b><br>
                Tanaman obat antidiabetes<br><br>

                <b>Tingkat kepercayaan sistem</b><br>
                {conf*100:.2f}%
            </div>
        """, unsafe_allow_html=True)

    # ---- Informasi ----
    st.markdown("""
        <div style="background:#f2f2f2; padding:20px; border-radius:8px; margin-top:20px;">
            <b>Informasi (jika herbal antidiabetes)</b><br>
            Tambahkan informasi herbal di sini...
        </div>
    """, unsafe_allow_html=True)

    # ---- Link Artikel ----
    st.text_input("Tautan artikel", "https://contoh-artikel.com")
    st.text_input("Tautan jurnal penelitian", "https://contoh-jurnal.com")

    # ---- Cara mengolah ----
    st.markdown("""
        <div style="background:#f2f2f2; padding:20px; border-radius:8px; margin-top:20px;">
            <b>Cara mengolah herbal antidiabetes</b><br>
            1. langkah 1<br>
            2. langkah 2<br>
            3. langkah 3<br>
            4. langkah x<br>
        </div>
    """, unsafe_allow_html=True)

    st.button("Kembali", on_click=lambda: (st.session_state.update({"page": "upload"}), st.rerun()))


# ---- FOOTER ----
st.markdown("<hr><center>Footer</center>", unsafe_allow_html=True)
