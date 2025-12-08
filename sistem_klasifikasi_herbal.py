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
        # CUSTOM CSS – UBAH FILE UPLOADER JADI KOTAK BESAR KEREN
        st.markdown("""
        <style>
        /* Hilangkan border default */
        [data-testid="stFileUploader"] section {
            border: 3px dashed #999 !important;
            padding: 60px !important;
            border-radius: 20px !important;
            background: #fafafa;
        }
        
        /* Ubah icon kecil jadi besar */
        [data-testid="stFileUploader"] section > div {
            text-align:center !important;
        }
        
        /* Hilangkan tulisan "Drag and drop" bawaan */
        [data-testid="stFileUploader"] label {
            display:none !important;
        }
        
        /* Teks custom */
        .custom-upload-text {
            text-align:center;
            font-size:22px;
            font-weight:600;
            margin-top:10px;
        }
        
        .custom-upload-sub {
            text-align:center;
            margin-top:-5px;
            color:#777;
        }
        </style>
        """, unsafe_allow_html=True)
        
        
        # =============================
        # FILE UPLOADER 
        # =============================
        uploaded_img = st.file_uploader("", type=["jpg","jpeg","png"])
        
        # PREVIEW GAMBAR
        if uploaded_img:
            img = Image.open(uploaded_img)
            st.markdown("##### 📌 Preview Gambar:")
            st.image(img, width=450)
        
        # =============================
        # TOMBOL IDENTIFIKASI
        # =============================
        if st.button("🔍 Identifikasi Daun", use_container_width=True):
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
                    <li>Pastikan helai daun berada tepat di tengah frame kamera</li>
                    <li>Pastikan pencahayaan mencukupi untuk dapat melihat venasi/urat daun</li>
                    <li>Latar belakang wajib polos dan berwarna cerah (diutamakan putih)</li>
                    <li>Fokus gambar daun jangan terlalu kecil</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


# =======================
# === HALAMAN HASIL =====
# =======================
elif st.session_state.page == "result":

    st.markdown("""
        <div style="text-align:center; margin-top:10px;">
            <h2 style="margin:0; font-size:55px; font-weight:600;">Hasil Identifikasi</h2>
        </div>
    """, unsafe_allow_html=True)

    img = Image.open(st.session_state.image)
    pred_name, conf = predict(img)

    colA, colB = st.columns([1.5,1])

    # ---- KIRI: Gambar & Detail ----
    with colA:
        colA1, colA2 = st.columns([1, 1.2])  # kiri gambar, kanan info
    
        with colA1:
            st.image(img, caption="Gambar yang diunggah", use_column_width=True)
    
        with colA2:
            st.markdown(f"""
                <div style="background:#ededed; padding:18px; border-radius:10px; margin-top:0px;">
                    <b>Nama Ilmiah</b><br>
                    {pred_name}<br>
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
                Tanaman Herbal Antidiabetes<br>
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
    st.markdown("""
        <div style="font-size:18px; font-weight:600; margin-top:25px; margin-bottom:8px;">
            Tautan artikel
        </div>
    """, unsafe_allow_html=True)
    st.text_input("", "https://contoh-artikel.com")
    
    st.markdown("""
        <div style="font-size:18px; font-weight:600; margin-top:15px; margin-bottom:8px;">
            Tautan jurnal penelitian
        </div>
    """, unsafe_allow_html=True)
    st.text_input("", "https://contoh-jurnal.com")

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

    # Tambah jarak sebelum tombol kembali
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    
    # Tombol kembali
    st.button("⬅️ Kembali", on_click=lambda: (st.session_state.update({"page": "upload"}), st.rerun()))


# ---- FOOTER ----
st.markdown("""
<style>
.footer-fixed {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: white;
    text-align: center;
    padding: 10px 0;
    font-size: 14px;
    border-top: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='footer-fixed'>©2025 | Klasifikasi Herbal Antidiabetes Berbasis Model LeafNet | 211401034 | Listy Zulmi</div>",
    unsafe_allow_html=True
)
