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
        "nama_umum": ["Bidara", "Widara", "Bukol", "Kalangga"],
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
        [data-testid="stFileUploader"] section {
            border: 3px dashed #999 !important;
            padding: 60px !important;
            border-radius: 20px !important;
            background: #fafafa;
        }
        /* Pastikan kolom kiri & kanan sejajar di atas */
        div[data-testid="column"] > div {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        </style>
        """, unsafe_allow_html=True)
        
        uploaded_img = st.file_uploader("", type=["jpg","jpeg","png"])
        
        # PREVIEW GAMBAR
        if uploaded_img:
            img = Image.open(uploaded_img)
            st.markdown("##### 📌 Preview Gambar:")
            st.image(img, width=320)
        
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
                    <li>Pastikan pencahayaan mencukupi agar model dapat melihat venasi/urat daun</li>
                    <li>Latar belakang daun wajib polos dan berwarna cerah (diutamakan putih)</li>
                    <li>Fokus gambar daun jangan terlalu kecil</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

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
                background: #ededed;
                padding: 18px;
                border-radius: 10px;
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
            st.markdown(f"""
                <div class="info-box">
                    <b class='section-title'>Nama Ilmiah:</b><br><b style="font-size:40px; font-weight:300;"><i>{pred_name}</i></b><br><br>
                    <b class='section-title'>Nama Umum:</b>
                """, unsafe_allow_html=True)
    
            # Menampilkan nama umum TANPA membuat HTML list baru
            if data and "nama_umum" in data:
                for nm in data["nama_umum"]:
                    st.write(f"- {nm}")
            else:
                st.write("- Tidak tersedia")
    
            # Tutup box
            st.markdown("</div>", unsafe_allow_html=True)

    # =====================================
    # KOLOM B — STATUS + CONFIDENCE (KANAN)
    # =====================================
    with colB:
        st.markdown(f"""
            <div style="background:#ededed; padding:18px; border-radius:10px;">
                <b class='section-title'>Status</b><br>
                <b style="color:#018790; font-weight:400;">{data['status'] if data else "Bukan herbal antidiabetes"}</b><br><br>
                <b>Tingkat kepercayaan sistem: </b> 
                <b style="color:#018790;">{conf * 100:.2f}% </b>
            </div>
        """, unsafe_allow_html=True)

    # === Informasi ===
    st.markdown("<div class='section-title'>Informasi</div>", unsafe_allow_html=True)
    st.markdown(data["informasi"] if data else "Tidak ada informasi.", unsafe_allow_html=True)

    # === Tautan Artikel ===
    st.markdown("<div class='section-title'>Tautan ke Artikel Terkait: </div>", unsafe_allow_html=True)
    st.markdown(
        f"<a href='{data['tautan_artikel']}' target='_blank'>{data['tautan_artikel']}</a>"
        if data else "Tidak ada link.",
        unsafe_allow_html=True
    )

    # === Tautan Jurnal Penelitian ===
    st.markdown("<div class='section-title'>Tautan ke Jurnal Penelitian</div>", unsafe_allow_html=True)
    st.markdown(
        f"<a href='{data['tautan_jurnal']}' target='_blank'>{data['tautan_jurnal']}</a>"
        if data else "Tidak ada link.",
        unsafe_allow_html=True
    )

    # === Cara Mengolah ===
    st.markdown("<div class='section-title', style='margin-bottom:10px;'>Cara Mengolah</div>", unsafe_allow_html=True)
    if data:
        for langkah in data["cara_mengolah"]:
            st.markdown(f"- {langkah}")
    else:
        st.markdown("- Tidak ada data.")

    # Jarak & tombol kembali
    st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
    st.button("⬅️ Kembali", on_click=lambda: (st.session_state.update({"page": "upload"}), st.rerun()))

# ---- FOOTER ----
st.markdown("""
<style>
main > div { padding-bottom: 0 !important; margin-bottom: 0 !important; }
.custom-footer { margin: 0 !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-footer">
    <hr>
    <center>©2025 | Klasifikasi Herbal Antidiabetes Berbasis Model LeafNet | 211401034 | Listy Zulmi</center>
</div>
""", unsafe_allow_html=True)
