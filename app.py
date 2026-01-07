import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Konfigurasi halaman
st.set_page_config(page_title="Laporan UAS PCD", layout="wide")

st.title("🚗 Implementasi PCD - Deteksi Tepi Mobil")

# --- PARAMETER HASIL ---
st.sidebar.header("⚙️ Parameter")
blur_val = st.sidebar.slider("Reduksi Noise (Blur)", 1, 15, 7, step=2)
low_thr = st.sidebar.slider("Canny Low Threshold", 0, 255, 50)
high_thr = st.sidebar.slider("Canny High Threshold", 0, 255, 150)
filter_size = st.sidebar.slider("Filter Objek (Hapus Orang)", 0, 1000, 400)

uploaded_file = st.file_uploader("Upload Gambar Mobil...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # --- LOAD GAMBAR ---
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- PRE-PROCESS ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
    edges = cv2.Canny(blurred, low_thr, high_thr)

    # --- FILTER KONTUR PANJANG (HAPUS ORANG) ---
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_edges = np.zeros_like(edges)

    for cnt in contours:
        if cv2.arcLength(cnt, True) > filter_size:
            # GARIS DIPERTEBAL (tebal=3)
            cv2.drawContours(filtered_edges, [cnt], -1, (255), 3)

    # --- INVERT WARNA AGAR GARIS HITAM DI LATAR PUTIH ---
    result_img = cv2.bitwise_not(filtered_edges)

    # --- KONVERSI GRAY KE RGB UNTUK OVERLAY ---
    result_color = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)

    # --- TRANSPARANSI OVERLAY ---
    alpha = 0.55
    combined = cv2.addWeighted(img_rgb, alpha, result_color, 1 - alpha, 0)

    # --- TAMPILAN TEMPLATE ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align: center;'>Gambar</h3>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.imshow(img_rgb)
        ax1.set_title("Gambar Asli")
        st.pyplot(fig1)

    with col2:
        st.markdown("<h3 style='text-align: center;'>Hasil Penerapan</h3>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.imshow(combined)
        ax2.set_title("HASIL DETEKSI TEPI DENGAN CANNY & FILTER KONTUR")
        st.pyplot(fig2)

    # --- DOWNLOAD ---
    plt.figure(figsize=(5, 4))
    plt.imshow(combined)
    plt.axis("off")
    plt.savefig("hasil_laporan.png", bbox_inches="tight")

    with open("hasil_laporan.png", "rb") as file:
        st.download_button("Simpan Gambar Hasil", file, "hasil_laporan.png", "image/png")

else:
    st.warning("Silakan upload gambar mobil terlebih dahulu.")
