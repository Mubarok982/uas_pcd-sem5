import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Deteksi Tepi Mobil",
    page_icon="🚗",
    layout="centered"
)

# --- Judul dan Deskripsi ---
st.title("🚗 Deteksi Tepi Kerusakan Mobil")
st.write("""
Aplikasi ini menggunakan metode **Canny Edge Detection** untuk memvisualisasikan garis tepi pada gambar mobil. 
Berguna untuk melihat tekstur penyok atau goresan dengan kontras tinggi.
""")

st.markdown("---")

# --- 1. Upload Gambar ---
uploaded_file = st.file_uploader("Unggah gambar mobil (JPG/PNG)...", type=["jpg", "jpeg", "png"])

# --- Sidebar: Pengaturan ---
st.sidebar.header("🔧 Pengaturan Sensitivitas")
st.sidebar.write("Geser untuk mengatur detail garis:")

# Slider untuk Threshold Canny
# Threshold rendah: Menangkap garis-garis halus (bisa jadi noise)
# Threshold tinggi: Hanya menangkap garis yang sangat tegas
thresh1 = st.sidebar.slider("Threshold Bawah (Detail)", min_value=0, max_value=255, value=100)
thresh2 = st.sidebar.slider("Threshold Atas (Ketegasan)", min_value=0, max_value=255, value=200)

st.sidebar.markdown("---")
st.sidebar.info("Tips: Jika gambar terlalu 'ramai' atau kotor, naikkan nilai Threshold Bawah.")

# --- Proses Utama ---
if uploaded_file is not None:
    # Baca file gambar
    image = Image.open(uploaded_file)
    # Konversi ke array numpy agar bisa dibaca OpenCV
    img_array = np.array(image)

    # Buat 2 Kolom untuk perbandingan
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Gambar Asli")
        st.image(image, use_container_width=True)

    # --- PROSES CANNY ---
    # 1. Ubah ke Grayscale (Hitam Putih) - Wajib untuk Canny
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 2. Terapkan Gaussian Blur (Opsional, untuk mengurangi noise bintik-bintik)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # 3. Algoritma Canny
    edges = cv2.Canny(blur_img, thresh1, thresh2)

    with col2:
        st.subheader("✏️ Hasil Deteksi Tepi")
        # Tampilkan hasil (edges adalah gambar hitam putih)
        st.image(edges, use_container_width=True, channels="GRAY")

    # --- Tombol Download Hasil ---
    # Konversi hasil array numpy kembali ke format gambar byte untuk didownload
    result_image = Image.fromarray(edges)
    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.markdown("###")
    st.download_button(
        label="⬇️ Download Hasil Deteksi Tepi",
        data=byte_im,
        file_name="hasil_deteksi_tepi.png",
        mime="image/png"
    )

else:
    # Tampilan awal jika belum upload
    st.info("Silakan unggah gambar mobil untuk memulai.")