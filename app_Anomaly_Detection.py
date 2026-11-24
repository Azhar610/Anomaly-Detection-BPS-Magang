import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from io import BytesIO

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Anomali Tabel BPS", layout="wide")

st.title("🔍 Asisten Validasi & Deteksi Anomali Tabel BPS")
st.markdown("""
Aplikasi ini menggunakan algoritma **Machine Learning (Isolation Forest)** untuk mendeteksi angka mencurigakan, 
salah ketik (*typo*), atau rasio yang tidak wajar antar-kolom pada draf tabel **Kapanewon/Kabupaten Dalam Angka**.
""")

# --- FUNGSI DATA CONTOH (DEMO KULON PROGO) ---
def load_sample_data():
    data = {
        "Kalurahan": [
            "Demangrejo", "Wijimulyo", "Bojong", "Sukoreno", "Brosot", 
            "Kaligintung", "Sidoharjo (Typo Nol)", "Glagah (Rasio Janggal)", 
            "giripeni", "Karangsari", "Jumlah"
        ],
        "Kapanewon": [
            "Sentolo", "Nanggulan", "Panjatan", "Sentolo", "Galur", 
            "Temon", "Samigaluh", "Temon", "Wates", "Pengasih", "-"
        ],
        "Luas Panen Padi (Ha)": [210, 340, 280, 250, 310, 230, 185, 95, 290, 305, 2495],
        "Produksi Padi (Ton)": [1320, 2150, 1750, 1550, 1920, 1420, 11400, 580, 1810, 1890, 25790],
        "Jumlah Kelompok Tani": [14, 18, 16, 15, 17, 14, 12, 85, 16, 17, 224]
    }
    return pd.DataFrame(data)

# --- SIDEBAR PENGATURAN ---
st.sidebar.header("⚙️ Pengaturan Sumber Data")
data_source = st.sidebar.radio(
    "Pilih Sumber Data:",
    ("Gunakan Data Contoh (Kulon Progo)", "Unggah File Excel (.xlsx)")
)

df_raw = None

if data_source == "Unggah File Excel (.xlsx)":
    uploaded_file = st.sidebar.file_uploader("Unggah Draf Tabel BPS (.xlsx)", type=["xlsx", "xls"])
    if uploaded_file:
        # Membaca daftar sheet jika file Excel memiliki banyak sheet
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = st.sidebar.selectbox("Pilih Sheet:", excel_file.sheet_names)
        skip_rows = st.sidebar.number_input("Lewati Baris Judul/Header Atas (Skip Rows):", min_value=0, max_value=20, value=0)
        df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=skip_rows)
else:
    df_raw = load_sample_data()
    st.sidebar.info("💡 Menggunakan data simulasi 10 Kalurahan di Kulon Progo (dengan 2 baris anomali buatan dan 1 baris 'Jumlah').")

# --- PROSES UTAMA ---
if df_raw is not None:
    st.subheader("1. Pratinjau Data Awal")
    st.dataframe(df_raw, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Konfigurasi Deteksi Machine Learning")

    col1, col2 = st.columns(2)

    with col1:
        # Memilih kolom identitas (misal: Nama Kalurahan)
        id_col = st.selectbox("Pilih Kolom Identitas Wilayah (Kalurahan/Kapanewon):", df_raw.columns, index=0)
        
        # Filter otomatis untuk membuang baris 'Jumlah' atau 'Total' khas tabel BPS
        remove_total = st.checkbox("Otomatis abaikan baris 'Jumlah / Total / Kabupaten'", value=True)

    # Pembersihan awal karakteristik tabel BPS (mengubah '-' atau '...' menjadi 0/NaN)
    df_clean = df_raw.copy()
    if remove_total:
        kata_kunci_total = ['jumlah', 'total', 'kabupaten', 'kapanewon', 'kecamatan', 'rata-rata']
        mask_total = df_clean[id_col].astype(str).str.lower().str.contains('|'.join(kata_kunci_total), na=False)
        df_total_rows = df_clean[mask_total]
        df_clean = df_clean[~mask_total].reset_index(drop=True)

    # Deteksi kolom yang berpotensi numerik (setelah mengganti tanda '-' khas BPS dengan 0)
    potential_num_cols = []
    for col in df_clean.columns:
        if col != id_col:
            cleaned_series = df_clean[col].astype(str).str.replace('-', '0').str.replace('...', '0').str.replace(',', '').str.strip()
            converted = pd.to_numeric(cleaned_series, errors='coerce')
            if converted.notna().sum() > (len(df_clean) * 0.5): # Jika >50% isinya angka
                df_clean[col] = converted.fillna(0)
                potential_num_cols.append(col)

    with col2:
        feature_cols = st.multiselect(
            "Pilih Kolom Angka yang Akan Diperiksa Silang (Minimal 2 Kolom):",
            options=potential_num_cols,
            default=potential_num_cols
        )
        
        contamination = st.slider(
            "Tingkat Sensitivitas / Perkiraan Persentase Error (Contamination):",
            min_value=0.01, max_value=0.30, value=0.15, step=0.01,
            help="Jika diisi 0.15, model akan mencari sekitar 15% baris yang polanya paling menyimpang."
        )

    # --- EKSEKUSI ISOLATION FOREST ---
    if len(feature_cols) >= 1:
        if st.button("🚀 Jalankan Deteksi Anomali", type="primary"):
            # Melatih model Isolation Forest
            model = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42
            )
            
            X = df_clean[feature_cols]
            preds = model.fit_predict(X)
            scores = model.decision_function(X) # Semakin negatif, semakin anomali

            # Menambahkan hasil ke tabel
            df_result = df_clean.copy()
            df_result["Status Validasi"] = np.where(preds == -1, "⚠️ ANOMALI", "✅ Normal")
            df_result["Skor Kejanggalan"] = np.round(scores, 3)

            # Urutkan agar baris Anomali muncul paling atas jika diinginkan, atau biarkan sesuai urutan asli
            jml_anomali = (preds == -1).sum()

            st.markdown("---")
            st.subheader(f"3. Hasil Deteksi: Ditemukan {jml_anomali} Baris Mencurigakan")

            # Fungsi pewarnaan merah untuk baris anomali
            def highlight_anomaly(row):
                if row["Status Validasi"] == "⚠️ ANOMALI":
                    return ['background-color: #ffcccc; color: #990000; font-weight: bold'] * len(row)
                return [''] * len(row)

            # Tampilkan di Streamlit dengan warna merah
            styled_df = df_result.style.apply(highlight_anomaly, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)

            # Menampilkan catatan baris 'Jumlah' yang tadi dipisahkan agar staf BPS tahu datanya tidak hilang
            if remove_total and not df_total_rows.empty:
                st.caption(f"ℹ️ Catatan: {len(df_total_rows)} baris rekapitulasi ('{', '.join(df_total_rows[id_col].astype(str))}') sengaja dipisahkan dari perhitungan ML agar tidak dianggap sebagai wilayah anomali.")

            # --- FITUR UNDUH EXCEL BERWARNA ---
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                styled_df.to_excel(writer, index=False, sheet_name='Hasil_Validasi_ML')
            
            st.download_button(
                label="📥 Unduh Hasil Validasi (Excel dengan Highlight Merah)",
                data=output.getvalue(),
                file_name="Hasil_Validasi_Anomali_BPS.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("Pilih minimal 1 kolom numerik untuk memulai deteksi.")