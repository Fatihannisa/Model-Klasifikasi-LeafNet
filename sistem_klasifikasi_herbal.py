import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit.components.v1 as components
import base64

def load_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

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
# ----- DATABASE DINAMIS --------
# =========================
herbal_info = {
    "Andrographis paniculata": {
        "nama_umum": ["Sambiloto", "Ki pait", "Ampadu tanah", "Ki oray"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": """
            Sambiloto terkenal sebagai herbal dengan kandungan andrographolide (AGL)
            yang sangat pahit, tetapi berkhasiat dalam mengendalikan kadar gula darah dan 
            bersifat antiinflamasi. AGL mampu meningkatkan produksi insulin dan  penyerapan 
            glukosa sehingga bisa mengurangi kadar gula dalam darah. 
        """,
        "tautan_artikel": "https://hellosehat.com/diabetes/daun-sambiloto-untuk-diabetes/",
        "tautan_jurnal": "https://jurnal.ikbis.ac.id/index.php/infokes/article/view/371/221 ",
        "cara_mengolah": [
            "Siapkan 25 lembar daun sambiloto dan 110 ml air.",
            "Cuci bersih daun sambiloto di bawah air mengalir.",
            "Rebus daun sambiloto sampai mendidih.",
            "Minum air rebusan daun sambiloto satu kali sehari dengan takaran 100 ml.",
            "Untuk menghindari risiko efek samping, disarankan untuk mengonsumsi dalam jumlah yang wajar dan tidak lebih dari dua kali sehari. Jika memiliki kondisi medis tertentu, konsultasikan terlebih dahulu dengan dokter sebelum mengonsumsi rebusan sambiloto.",
        ]
    },
    
    "Ziziphus mauritiana": {
        "nama_umum": ["Bidara", "Widara", "Bukol", "Kalangga", "Bekul", "Rangga"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": """
            Daun bidara bisa membantu mengendalikan diabetes dengan membuat penggunaan insulin 
            untuk menyerap gula darah lebih efektif. Kandungan saponin dan flavonoid di dalam 
            daun bidara bekerja sebagai antioksidan yang dapat membantu melawan stres oksidatif 
            akibat radikal bebas. Dengan begitu, konsumsi ekstrak daun bidara dapat mendukung 
            pencegahan diabetes, terutama jika disertai pola hidup sehat.
        """,
        "tautan_artikel": "https://hellosehat.com/herbal-alternatif/herbal/daun-bidara/",
        "tautan_jurnal": "https://doi.org/10.26740/icaj.v6i2.32598",
        "cara_mengolah": [
            "Siapkan 10 lembar daun bidara tua, setengah buah jeruk nipis, dan gula secukupnya.",
            "Cuci bersih daun sambiloto di bawah air mengalir.",
            "Rebus 600ml air hingga mendidih lalu masukkan daun bidara.",
            "Masak selama 20 menit dengan api kecil.",
            "Peras jeruk nipis. Tambahkan gula sesuai selera.",
            "Untuk menghindari risiko efek samping, disarankan untuk mengonsumsi dalam jumlah yang wajar. Jika memiliki kondisi medis tertentu, konsultasikan terlebih dahulu dengan dokter sebelum mengonsumsi rebusan ini.",
        ]
    },

    "Pandanus amaryllifolius": {
        "nama_umum": ["Pandan wangi", "Pandan", "Pandan rampe", "Pandan arrum"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": """
            Pandanus amaryllifolius adalah tanaman tropis yang umum dikenal sebagai pandan. Daun pandan mengandung senyawa seperti flavonoid, tanin, dan polifenol. Menurut sebuah studi dalam jurnal Pharmacognosy Magazine, ekstrak pandan mampu merangsang produksi hormon insulin dari pankreas.
        """,
        "tautan_artikel": "https://www.halodoc.com/artikel/manfaat-daun-pandan-dan-efek-sampingnya-bagi-tubuh?srsltid=AfmBOoq7fJ-Up5emjKX-zFlcgvMiiFUf-myDu9zotaOuc1uwwwfuYXun",
        "tautan_jurnal": "https://ejurnalmalahayati.ac.id/index.php/kebidanan/article/view/3024/pdf",
        "cara_mengolah": [
            "Siapkan 3-4 lembar daun pandan(segar atau kering), 500 ml air, dan pemanis alami(jika diperlukan).",
            "Cuci bersih daun pandan di bawah air mengalir lalu potong menjadi beberapa bagian.",
            "Rebus 500ml air hingga mendidih lalu masukkan potongan daun pandan.",
            "Biarkan daun pandan direbus selama 10-15 menit hingga air berubah warna menjadi hijau kekuningan.",
            "Saring air rebusan dan tuangkan ke dalam gelas (tambahkan pemanis alami jika diperlukan).",
            "Untuk menghindari risiko efek samping, disarankan untuk mengonsumsi dalam jumlah yang wajar. Jika memiliki kondisi medis tertentu, konsultasikan terlebih dahulu dengan dokter sebelum mengonsumsi rebusan ini.",
        ]
    },
}
    
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
st.set_page_config(page_title="Sistem Identifikasi Herbal Antidiabetes Berbasis LeafNet 🌿 ", layout="wide")

# --- GLOBAL ADAPTIVE CSS ---
st.markdown("""
<style>
/* ===============================
   VARIABEL WARNA ADAPTIF GLOBAL
   =============================== */
:root {
    /* Light mode */
    --bg-box: #f1f1f1;
    --border-box: #d0d0d0;

    --bg-uploader: #f8f8f8;
    --border-uploader: #bbb;
}

@media (prefers-color-scheme: dark) {
    :root {
        /* Dark mode */
        --bg-box: #2a2a2a;
        --border-box: #444;

        --bg-uploader: #1f1f1f;
        --border-uploader: #555;
    }
}

/* ===============================
   INFO BOX (nama ilmiah/umum)
   =============================== */
.info-box, .adaptive-box {
    background: var(--bg-box) !important;
    border: 1px solid var(--border-box) !important;
    padding: 18px;
    border-radius: 12px;
    color: inherit !important;
}

/* ===============================
   FILE UPLOADER BOX
   =============================== */
[data-testid="stFileUploader"] section {
    background: var(--bg-uploader) !important;
    border: 3px dashed var(--border-uploader) !important;
    padding: 60px !important;
    border-radius: 20px !important;
    min-height: 260px !important;
}

/* Agar teks + ikon uploader terlihat */
[data-testid="stFileUploader"] * {
    color: inherit !important;
}

/* Kolom tetap rata atas */
div[data-testid="column"] > div {
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

/* Section title */
.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-top: 25px;
}

/* FOOTER ADAPTIF */
.custom-footer {
    padding: 12px 0 20px 0;
    text-align: center;
    border-radius: 10px;
}

/* Light Mode */
@media (prefers-color-scheme: light) {
    .custom-footer {
        background: #f2f2f2 !important;
        color: #000 !important;
    }
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
    .custom-footer {
        background: #1e1e1e !important;
        color: #eaeaea !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ---- HEADER ----
logo_base64 = load_base64("images/diaherb_logo.png")
components.html(f"""
    <div style="
        padding:0 20px; 
        width:100%; 
        display:flex; 
        align-items:center;
    ">
        <img src="data:image/png;base64,{logo_base64}"
             style="height:100px; width:auto;
             filter: drop-shadow(0px 0px 4px rgba(0,0,0,0.35));">
    </div>
    <hr>
""", height=100)


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
        <h2 style="margin:0px; font-size:45px; font-weight:600;">Sistem Identifikasi Daun Herbal Antidiabetes Berbasis Model LeafNet</h2>
        <p style="font-size:18px; margin-bottom:40px; width:95%;">DiaHerb merupakan sebuah 
        sistem berbasis teknologi yang dikembangkan untuk mengidentifikasi tanaman herbal 
        antidiabetes berdasarkan citra daun. DiaHerb bertujuan mengidentifikasi tanaman herbal 
        antidiabetes secara tepat untuk mendukung penelitian dan edukasi masyarakat, serta 
        mempromosikan potensi tanaman herbal lokal Indonesia. Sistem ini dibangun berbasis Deep 
        Learning menggunakan Model LeafNet yang diintegrasikan dengan Transfer Learning untuk menganalisis ciri pada daun.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.6,1.2])

    with col1:
        # CUSTOM CSS – UBAH FILE UPLOADER JADI KOTAK BESAR
        st.markdown("""
        <style>
        [data-testid="stFileUploader"] section {
            border: 3px dashed #999 !important;
            padding: 60px !important;
            border-radius: 20px !important;
            background: #fafafa;
            min-height: 260px !important;
        }
        /* Pastikan kolom kiri & kanan sejajar di atas */
        div[data-testid="column"] > div {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        </style>
        """, unsafe_allow_html=True)
        
        uploaded_img = st.file_uploader("", type=["jpg","jpeg","png", "webp"])

        img = None 
        
        # PREVIEW GAMBAR
        if uploaded_img is not None:
            try:
                img = Image.open(uploaded_img)
                st.markdown("##### 📌 Preview Gambar:")
                st.image(img, width=320)
            except Exception as e:
                st.error(f"Format gambar tidak dapat dibaca: {e}")
        
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

    # =============================
    # TIPS PENGAMBILAN GAMBAR
    # =============================
    st.markdown("""
    <style>
        /* Rapatkan jarak antar gambar dalam kolom */
        div[data-testid="column"] div:has(img) {
            padding-right: 5px !important;
            padding-left: 5px !important;
            margin-right: 0 !important;
            margin-left: 0 !important;
        }
    
        /* Hilangkan margin top berlebih pada gambar */
        div[data-testid="column"] div:has(img) {
            margin-top: 5px !important;
        }
    
    </style>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <b style="font-size:18px; font-weight:600; margin-left:30px;">Tips pengambilan gambar</b>
            <ul style="margin-left:30px;">
                <li>Foto 1 helai daun saja.</li>
                <li>Pastikan helai daun berada tepat di tengah frame kamera.</li>
                <li>Pastikan pencahayaan mencukupi agar model dapat melihat venasi/urat daun.</li>
                <li>Latar belakang daun wajib polos dan berwarna cerah (diutamakan putih).</li>
                <li>Foto objek tidak terlalu jauh.</li>
                <li>Foto dari sisi atas atau bawah helai daun.</li>
            </ul>
    
            <b style="font-size:18px; font-weight:600; margin-left:30px;">Contoh gambar yang baik</b>
        """, unsafe_allow_html=True)

        # Gambar contoh: 3 atau 4
        offset_col, *ex_cols = st.columns([0.2, 1, 1, 1, 1])
        
        example_paths = [
            "images/IMG_20251028_152831.jpg",
            "images/IMG_20251029_170845.jpg",
            "images/IMG_20251031_131056.jpg",
            "images/IMG_20251114_161441.jpg"
        ]
        
        for col, path in zip(ex_cols, example_paths):
            with col:
                st.image(path, width=150)

# =======================
# === HALAMAN HASIL =====
# =======================
elif st.session_state.page == "result":

    st.markdown("""
        <h2 style="margin-bottom:30px; font-size:50px; font-weight:600; text-align:center;">
            Hasil Identifikasi
        </h2>
    """, unsafe_allow_html=True)

    img = Image.open(st.session_state.image)
    pred_name, conf = predict(img)

    # ambil data dari database
    data = herbal_info.get(pred_name, None)

    colA, colB = st.columns([1.5,1])

    # =====================================
    # KOLOM A — GAMBAR + INFO BOX (KIRI)
    # =====================================
    with colA:
        colA1, colA2 = st.columns([1, 1.2])
        
        # --- KIRI: Gambar ---
        with colA1:
            st.image(img, caption="Gambar yang diunggah", use_column_width=True)

        # --- CSS untuk info box ---
        st.markdown("""
            <style>
            .info-box {
                background: var(--bg-box);
                border: 1px solid var(--border-box);
                padding: 18px;
                border-radius: 12px;
                min-height: 340px;
            }
            .section-title {
                font-size: 20px;
                font-weight: 600;
                margin-top: 25px;
            }
            </style>
        """, unsafe_allow_html=True)

        # --- KANAN: Box Info ---
        with colA2:
            # buat list nama umum dalam HTML
            if data and "nama_umum" in data:
                list_html = "<ul style='margin-top:5px;'>"
                for nm in data["nama_umum"]:
                    list_html += f"<li>{nm}</li>"
                list_html += "</ul>"
            else:
                list_html = "<ul><li>Tidak tersedia</li></ul>"
        
            # HTML box lengkap
            html_box = f"""
                <div class="info-box">
                    <b class="section-title">Nama Ilmiah:</b><br>
                    <span style="font-size:40px; font-weight:300;"><i>{pred_name}</i></span>
                    <br><br>
                    <b class="section-title">Nama Umum:</b>
                    {list_html}
                </div>
            """
            st.markdown(html_box, unsafe_allow_html=True)

    # =====================================
    # KOLOM B — STATUS + CONFIDENCE (KANAN)
    # =====================================
    with colB:
        st.markdown(f"""
            <div class="adaptive-box">
                <b class='section-title'>Status</b><br>
                <b style="color:#018790; font-weight:400;">{data['status'] if data else "Bukan herbal antidiabetes"}</b><br><br>
                <b>Tingkat kepercayaan sistem: </b> 
                <b style="color:#018790;">{conf * 100:.2f}% </b>
            </div>
        """, unsafe_allow_html=True)

    # === Informasi ===
    st.markdown("<div class='section-title'>Informasi herbal:</div>", unsafe_allow_html=True)
    st.markdown(data["informasi"] if data else "Tidak ada informasi.", unsafe_allow_html=True)

    # === Tautan Artikel ===
    st.markdown("<div class='section-title'>Tautan ke artikel terkait: </div>", unsafe_allow_html=True)
    st.markdown(
        f"<a href='{data['tautan_artikel']}' target='_blank'>{data['tautan_artikel']}</a>"
        if data else "Tidak ada link.",
        unsafe_allow_html=True
    )

    # === Tautan Jurnal Penelitian ===
    st.markdown("<div class='section-title'>Tautan ke jurnal penelitian:</div>", unsafe_allow_html=True)
    st.markdown(
        f"<a href='{data['tautan_jurnal']}' target='_blank'>{data['tautan_jurnal']}</a>"
        if data else "Tidak ada link.",
        unsafe_allow_html=True
    )

    # === Cara Mengolah ===
    st.markdown("<div class='section-title', style='margin-bottom:10px;'>Cara mengolah herbal:</div>", unsafe_allow_html=True)
    if data:
        for langkah in data["cara_mengolah"]:
            st.markdown(f"- {langkah}")
    else:
        st.markdown("- Tidak ada data.")

    # Jarak & tombol kembali
    st.markdown("<div style='height:70px;'></div>", unsafe_allow_html=True)
    st.button("⬅️ Kembali", on_click=lambda: (st.session_state.update({"page": "upload"}), st.rerun()))


st.markdown("""
<style>
/* DISCLAIMER BOX ADAPTIVE */
.disclaimer-box {
    padding: 12px 18px;
    border-radius: 10px;
    margin-top: 30px;
    font-size: 15px;
    line-height: 1.45;
}

/* Light Mode */
@media (prefers-color-scheme: light) {
    .disclaimer-box {
        background: #f8f8f8;
        color: #333;
        border: 1px solid #ddd;
    }
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
    .disclaimer-box {
        background: #2a2a2a;
        color: #e2e2e2;
        border: 1px solid #444;
    }
}
</style>

<div class="disclaimer-box">
    <strong>Disclaimer:</strong><br>
    Sistem klasifikasi herbal ini dikembangkan sebagai bagian dari penyusunan tugas akhir.
    Hasil prediksi bersifat estimasi dan tidak dimaksudkan sebagai acuan medis atau botani yang bersifat final.
    Validasi tetap disarankan melalui literatur ilmiah atau ahli terkait.
</div>
""", unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("""
<div class="custom-footer">
    <hr>
    ©2025 | Klasifikasi Herbal Antidiabetes Berbasis Model LeafNet | 211401034 | Listy Zulmi
</div>
""", unsafe_allow_html=True)
