import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================
# CONFIG
# =====================
st.set_page_config(
    page_title="Ketimpangan Pendidikan Kota Bandung",
    layout="wide",
    # BARIS KUNCI:
    initial_sidebar_state="collapsed" 
)

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    # Asumsi file 'df_eda.csv' dan 'dataset_final.csv' ada di direktori yang sama
    try:
        df_eda = pd.read_csv("df_eda.csv")
        dfp = pd.read_csv("dataset_final.csv", index_col=0)
    except FileNotFoundError:
        st.error("Pastikan file 'df_eda.csv' dan 'dataset_final.csv' tersedia.")
        # Membuat dataframe dummy agar kode selanjutnya tidak error
        df_eda = pd.DataFrame({
            "bps_desa_kelurahan": ["Kelurahan A", "Kelurahan B"], 
            "rendah_pct": [0.3, 0.6], 
            "menengah_pct": [0.4, 0.3], 
            "tinggi_pct": [0.3, 0.1]
        })
        dfp = df_eda.set_index("bps_desa_kelurahan").copy()
        
    return df_eda, dfp

df_eda, dfp = load_data()

# =======================================================
# SESSION STATE & NAVIGASI KARTU (DI AREA UTAMA)
# =======================================================

# Inisialisasi Session State untuk navigasi
if 'current_menu' not in st.session_state:
    st.session_state['current_menu'] = "🏠 Home"

menu_list = [
    "🏠 Home",
    "📦 Data Preparation",
    "🔎 Exploratory Data Analysis (EDA)",
    "⚙️ Preprocessing",
    "🎯 K-Means Clustering"
]

# Ambil menu aktif dari session state
menu = st.session_state['current_menu']

# =====================
# SIDEBAR (STATIC & CLEANED)
# =====================
# Hanya menampilkan judul statis di sidebar
st.sidebar.markdown(
    """
    <h2 style='text-align: center;'>🧭 Alur Analisis</h2>
    <p style='text-align: center; font-size: 14px;'>
    Identifikasi Ketimpangan Pendidikan<br/>
    Kota Bandung
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
# Menampilkan menu aktif (opsional)
st.sidebar.markdown(f"**Menu Aktif:** {menu}")
st.sidebar.caption("Navigasi utama berada di bagian atas halaman.")


# =======================================================
# MAIN AREA NAVIGATION (CARD STYLE)
# =======================================================
st.markdown(
    """
    <h1 style='text-align: center;'>
    Identifikasi Ketimpangan Pendidikan Penduduk<br/>
    Antar Kelurahan di Kota Bandung
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("### 🗺️ Pilih Tahapan Analisis")

cols = st.columns(len(menu_list))

# Membuat Kartu Navigasi
for i, item in enumerate(menu_list):
    
    # Menentukan apakah item ini sedang aktif
    is_active = item == menu
    
    with cols[i]:
        # Tombol sebagai pemindah menu
        if st.button(
            item, # Tampilkan nama lengkap pada tombol
            key=f"nav_{item}",
            use_container_width=True,
        ):
            st.session_state['current_menu'] = item
            st.rerun()

st.markdown("---") # Garis pembatas visual

# =====================
# KONTEN BERDASARKAN MENU (SEMUA KODE KONTEN DI BAWAH INI TETAP SAMA)
# =====================
if menu == "🏠 Home":
    # KONTEN HOME
    st.markdown("""
        <p style='text-align: center; font-size: 18px;'>
        Pendekatan Data dan K-Means Clustering
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    Pendidikan merupakan salah satu indikator penting dalam pembangunan wilayah.
    Namun, tingkat pendidikan penduduk di Kota Bandung belum tersebar secara merata
    antar kelurahan.
    """)

    st.markdown("""
    Beberapa kelurahan memiliki akses pendidikan yang relatif baik hingga jenjang
    tinggi, sementara kelurahan lainnya masih didominasi oleh pendidikan rendah.
    Ketimpangan ini perlu dipahami agar kebijakan pendidikan dapat diarahkan
    secara lebih tepat sasaran.
    """)

    st.markdown("""
    Melalui pendekatan **analisis data** dan **K-Means Clustering**, studi ini
    bertujuan untuk mengelompokkan kelurahan berdasarkan kemiripan struktur
    pendidikan penduduk.
    """)

    st.success("""
    🎯 **Tujuan Utama Analisis**  
    Mengidentifikasi kelompok kelurahan dengan tingkat ketimpangan pendidikan
    yang berbeda sebagai dasar perumusan rekomendasi pemerataan pendidikan.
    """)

    st.markdown("---")

    st.subheader("🧭 Alur Analisis")
    st.markdown("""
    Analisis dilakukan melalui tahapan berikut:
    - **Data Preparation**: penyiapan dan pembentukan data pendidikan per kelurahan.
    - **EDA**: eksplorasi pola dan ketimpangan pendidikan antar wilayah.
    - **Preprocessing**: penyamaan skala data sebelum clustering.
    - **K-Means Clustering**: pengelompokan kelurahan dan interpretasi hasil.
    """)

    st.info(
        "Gunakan menu navigasi di bagian atas halaman untuk mengikuti alur analisis "
        "dari awal hingga akhir."
    )

# =====================
# DATA PREPARATION
# =====================
elif menu == "📦 Data Preparation":
    st.markdown(
    "<h1 style='text-align: center;'>Data Preparation</h1>",
    unsafe_allow_html=True
)
    # ... (Konten Data Preparation yang sudah Anda buat) ...
    st.markdown("""
    Tahap **Data Preparation** bertujuan untuk menjelaskan proses penyiapan data 
    pendidikan penduduk Kota Bandung yang digunakan dalam analisis ketimpangan 
    dan metode clustering.
    """)

    # =====================
    # TUJUAN
    # =====================
    st.subheader("📌 Tujuan Data Preparation")
    st.markdown("""
    - Menyusun data pendidikan penduduk pada level **kelurahan**.
    - Membentuk representasi kuantitatif untuk mengukur **ketimpangan pendidikan**.
    - Menyiapkan dataset yang siap digunakan pada tahap **EDA** dan **K-Means Clustering**.
    """)

    # =====================
    # SUMBER DATA
    # =====================
    st.subheader("🗂️ Sumber dan Cakupan Data")
    st.markdown("""
    - Data merupakan hasil agregasi penduduk berdasarkan **tingkat pendidikan**.
    - Setiap observasi merepresentasikan **satu kelurahan di Kota Bandung**.
    - Fokus analisis berada pada perbedaan struktur pendidikan antar kelurahan.
    """)

    # =====================
    # VARIABEL
    # =====================
    st.subheader("📊 Variabel yang Digunakan")
    st.markdown("""
    Untuk menggambarkan struktur pendidikan penduduk, digunakan tiga variabel utama:
    - **Pendidikan Rendah (`rendah_pct`)**: proporsi penduduk berpendidikan dasar.
    - **Pendidikan Menengah (`menengah_pct`)**: proporsi penduduk berpendidikan menengah.
    - **Pendidikan Tinggi (`tinggi_pct`)**: proporsi penduduk berpendidikan tinggi.

    Ketiga variabel ini membentuk komposisi pendidikan tiap kelurahan.
    """)

    # =====================
    # PENGOLAHAN DATA
    # =====================
    st.subheader("📖 Proses Pengolahan Data")
    st.markdown("""
    - Data pendidikan diringkas dalam bentuk **persentase** untuk setiap kelurahan (dari tahun 2017 hingga 2025).
    - Agregasi dilakukan agar perbandingan antar wilayah dapat dilakukan secara adil.
    - Dataset dipisahkan sesuai kebutuhan analisis, yaitu untuk:
        - **Exploratory Data Analysis (EDA)**
        - **Clustering menggunakan K-Means**
    """)

    # =====================
    # OUTPUT
    # =====================
    st.subheader("🎯 Output Tahap Data Preparation")
    st.markdown("""
    Hasil dari tahap ini adalah dataset yang:
    - Mewakili kondisi ketimpangan pendidikan antar kelurahan.
    - Siap digunakan untuk analisis eksploratif dan pemodelan clustering.
    - Menjadi dasar dalam proses pengambilan insight dan rekomendasi kebijakan.
    """)

    st.markdown("---")
    st.caption("Tahap selanjutnya akan membahas pola dan ketimpangan pendidikan melalui Exploratory Data Analysis (EDA).")


# =====================
# EDA
# =====================
elif menu == "🔎 Exploratory Data Analysis (EDA)":

    st.markdown(
        """
        <h1 style='text-align: center;'>Exploratory Data Analysis (EDA)</h1>
        <p style='text-align: center; font-size: 18px;'>
        Menggali pola dan ketimpangan pendidikan antar kelurahan
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    Tahap **Exploratory Data Analysis (EDA)** bertujuan untuk memahami pola,
    sebaran, dan hubungan antar tingkat pendidikan penduduk di setiap kelurahan
    sebelum dilakukan proses clustering.
    """)

    st.info(
        "Visualisasi berikut membantu mengidentifikasi kelurahan dengan tingkat "
        "ketimpangan pendidikan yang ekstrem, pola sebaran struktur pendidikan, "
        "serta hubungan antar tingkat pendidikan sebagai dasar pengelompokan."
    )

    # ==================================================
    # LINE PLOT – EXTREME VALUES
    # ==================================================
    st.subheader("Kelurahan dengan Ketimpangan Pendidikan Paling Ekstrem")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Top 10 kelurahan dengan proporsi pendidikan rendah tertinggi.")
        top10_rendah = df_eda.sort_values("rendah_pct", ascending=False).head(10)
        fig, ax = plt.subplots()
        ax.plot(
            top10_rendah["bps_desa_kelurahan"],
            top10_rendah["rendah_pct"],
            marker="o"
        )
        ax.set_xticklabels(top10_rendah["bps_desa_kelurahan"], rotation=45)
        ax.set_ylabel("Proporsi Pendidikan Rendah")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.write("Top 10 kelurahan dengan proporsi pendidikan tinggi tertinggi.")
        top10_tinggi = df_eda.sort_values("tinggi_pct", ascending=False).head(10)
        fig, ax = plt.subplots()
        ax.plot(
            top10_tinggi["bps_desa_kelurahan"],
            top10_tinggi["tinggi_pct"],
            marker="o"
        )
        ax.set_xticklabels(top10_tinggi["bps_desa_kelurahan"], rotation=45)
        ax.set_ylabel("Proporsi Pendidikan Tinggi")
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🔍 Apa yang perlu diperhatikan dari grafik ini?"):
        st.markdown("""
        - Terdapat perbedaan ekstrem antar kelurahan dalam proporsi pendidikan rendah dan tinggi.
        - Kelurahan dengan pendidikan rendah tinggi tidak selalu memiliki pendidikan tinggi yang besar.
        - Hal ini mengindikasikan adanya ketimpangan struktural antar wilayah.
        """)

    # ==================================================
    # BOXPLOT – DISTRIBUTION
    # ==================================================
    st.subheader("Seberapa Merata Struktur Pendidikan Antar Kelurahan?")
    col3, col4 = st.columns(2)

    with col3:
        st.write("Sebaran pendidikan rendah dan tinggi antar kelurahan.")
        fig, ax = plt.subplots()
        ax.boxplot(
            [df_eda["rendah_pct"], df_eda["tinggi_pct"]],
            labels=["Pendidikan Rendah", "Pendidikan Tinggi"]
        )
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        st.write("Struktur pendidikan lengkap per kelurahan.")
        fig, ax = plt.subplots()
        ax.boxplot(
            [
                df_eda["rendah_pct"],
                df_eda["menengah_pct"],
                df_eda["tinggi_pct"]
            ],
            labels=[
                "Pendidikan Rendah",
                "Pendidikan Menengah",
                "Pendidikan Tinggi"
            ]
        )
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🔍 Insight dari boxplot"):
        st.markdown("""
        - Sebaran nilai menunjukkan variasi yang cukup besar antar kelurahan.
        - Pendidikan rendah memiliki rentang yang lebih lebar dibanding pendidikan tinggi.
        - Pola ini menandakan ketimpangan pendidikan tidak terjadi secara merata.
        """)

    # ==================================================
    # SCATTER & CORRELATION
    # ==================================================
    st.subheader("Hubungan Antar Tingkat Pendidikan")

    st.markdown(
        "**Grafik berikut menunjukkan hubungan langsung antara pendidikan rendah dan "
        "pendidikan tinggi, yang menjadi dasar penting dalam proses clustering.**"
    )

    col5, col6 = st.columns(2)

    with col5:
        fig, ax = plt.subplots()
        ax.scatter(df_eda["rendah_pct"], df_eda["tinggi_pct"])
        ax.set_xlabel("Proporsi Pendidikan Rendah")
        ax.set_ylabel("Proporsi Pendidikan Tinggi")
        st.pyplot(fig)
        plt.close(fig)

    with col6:
        corr = df_eda[["rendah_pct", "menengah_pct", "tinggi_pct"]].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🔍 Interpretasi hubungan antar variabel"):
        st.markdown("""
        - Pendidikan rendah dan pendidikan tinggi menunjukkan hubungan negatif.
        - Kelurahan dengan pendidikan rendah tinggi cenderung memiliki pendidikan tinggi yang rendah.
        - Pola ini menunjukkan adanya kelompok kelurahan dengan karakteristik serupa.
        """)

    # ==================================================
    # KEY FINDINGS
    # ==================================================
    st.markdown("---")
    st.markdown(
        "Berdasarkan seluruh visualisasi di atas, berikut adalah temuan utama dari tahap EDA:"
    )

    st.markdown("""
    ### Key Findings EDA
    - Ketimpangan pendidikan antar kelurahan terlihat jelas dan tidak merata.
    - Terdapat kelurahan dengan dominasi pendidikan rendah maupun pendidikan tinggi.
    - Pendidikan rendah dan pendidikan tinggi memiliki hubungan negatif yang kuat.
    - Pola ketimpangan ini menjadi dasar perlunya pengelompokan kelurahan menggunakan metode clustering (K-Means).
    """)

    st.markdown("""
    ### Mengapa Perlu Clustering?
    Meskipun EDA telah menunjukkan adanya ketimpangan pendidikan dan pola hubungan
    antar tingkat pendidikan, visualisasi saja belum cukup untuk mengelompokkan
    kelurahan secara objektif dan konsisten.
    """)

    st.success("""
    Oleh karena itu, diperlukan metode **clustering** untuk mengelompokkan kelurahan
    berdasarkan kemiripan struktur pendidikan. Metode **K-Means** digunakan untuk
    mengidentifikasi kelompok kelurahan dengan karakteristik ketimpangan pendidikan
    yang serupa, sehingga analisis dapat dilanjutkan ke tahap interpretasi dan
    perumusan rekomendasi.
    """)

    st.caption("Tahap selanjutnya: pengelompokan kelurahan menggunakan metode K-Means Clustering.")


# =====================
# PREPROCESSING
# =====================
elif menu == "⚙️ Preprocessing":

    st.markdown(
        """
        <h1 style='text-align: center;'>Preprocessing Data</h1>
        <p style='text-align: center; font-size: 18px;'>
        Menyamakan skala data sebelum proses clustering
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    Sebelum dilakukan pengelompokan kelurahan menggunakan K-Means, 
    data perlu melalui tahap **preprocessing** agar perbandingan antar 
    kelurahan dilakukan secara adil dan seimbang.
    """)

    # ==================================================
    # WHY PREPROCESSING
    # ==================================================
    st.subheader("⚖️ Mengapa Preprocessing Diperlukan?")
    st.markdown("""
    - Setiap variabel pendidikan memiliki rentang nilai yang berbeda.
    - Tanpa preprocessing, variabel tertentu dapat mendominasi hasil clustering.
    - Preprocessing membantu memastikan setiap indikator memiliki kontribusi yang setara.
    """)

    # ==================================================
    # FEATURE OVERVIEW
    # ==================================================
    st.subheader("📊 Variabel yang Digunakan")
    st.dataframe(
        dfp[["rendah_pct", "menengah_pct", "tinggi_pct"]].head()
    )

    st.markdown("""
    Ketiga variabel di atas digunakan untuk merepresentasikan struktur pendidikan
    pada masing-masing kelurahan.
    """)

    # ==================================================
    # INTERACTIVE EXAMPLE
    # ==================================================
    st.subheader("🔍 Ilustrasi Dampak Preprocessing")

    sample = dfp[["rendah_pct", "menengah_pct", "tinggi_pct"]].head()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Nilai Asli (Sebelum Preprocessing)**")
        st.dataframe(sample)

    scaler = StandardScaler()
    sample_scaled = pd.DataFrame(
        scaler.fit_transform(sample),
        columns=sample.columns,
        index=sample.index
    )

    with col2:
        st.markdown("**Nilai Setelah Preprocessing (Standardisasi)**")
        st.dataframe(sample_scaled)

    st.markdown("""
    Perbandingan ini menunjukkan bahwa preprocessing menyamakan skala antar variabel,
    sehingga setiap indikator pendidikan memiliki kontribusi yang seimbang
    dalam proses clustering.
    """)


    # ==================================================
    # TAKEAWAY
    # ==================================================
    st.subheader("🧠 Inti Tahap Preprocessing")
    st.markdown("""
    - Preprocessing tidak mengubah pola dasar data.
    - Tujuannya adalah menyamakan skala agar hasil clustering lebih stabil.
    - Tahap ini menjadi fondasi penting sebelum proses K-Means Clustering.
    """)

# =====================
# K-MEANS
# =====================
elif menu == "🎯 K-Means Clustering":

    # ==================================================
    # HEADER
    # ==================================================
    st.markdown(
        """
        <h1 style='text-align: center;'>K-Means Clustering</h1>
        <p style='text-align: center; font-size: 18px;'>
        Pengelompokan kelurahan berdasarkan tingkat ketimpangan pendidikan
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    Halaman ini menyajikan hasil pengelompokan kelurahan di Kota Bandung 
    berdasarkan komposisi tingkat pendidikan penduduk menggunakan metode 
    **K-Means Clustering**.
    """)

    # ==================================================
    # PREPARE DATA (BACKGROUND)
    # ==================================================
    X = dfp[["rendah_pct", "menengah_pct", "tinggi_pct"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ensure n_clusters is not greater than number of samples
    n_clusters_default = 3
    if len(dfp) < n_clusters_default:
        n_clusters_default = len(dfp)
        
    try:
        kmeans = KMeans(n_clusters=n_clusters_default, random_state=42, n_init=10)
        dfp["cluster"] = kmeans.fit_predict(X_scaled)
    except ValueError:
        st.error("Gagal menjalankan K-Means. Mungkin data terlalu sedikit.")
        dfp["cluster"] = 0 # Default cluster jika gagal

    # ==================================================
    # 1. CEK KLASTER KELURAHAN (HERO SECTION)
    # ==================================================
    st.markdown("---")
    st.subheader("🎯 Cek Klaster Kelurahan")

    st.markdown("""
    Pilih nama kelurahan untuk mengetahui hasil pengelompokan, 
    karakteristik klaster, serta rekomendasi berdasarkan kondisi 
    ketimpangan pendidikan.
    """)

    kelurahan_list = sorted(dfp.index.tolist())

    selected_kelurahan = st.selectbox(
        "Pilih Kelurahan",
        kelurahan_list
    )

    cek = st.button("Cek Klaster")

    if cek and len(dfp) >= n_clusters_default:
        cluster_id = dfp.loc[selected_kelurahan, "cluster"]

        st.markdown(f"### Hasil untuk Kelurahan **{selected_kelurahan}**")
        st.write(f"Masuk ke **Klaster {cluster_id}**")

        if cluster_id == 0:
            st.markdown("""
            **Karakteristik Klaster 0 (Ketimpangan Rendah)**  
            Struktur pendidikan relatif seimbang dengan proporsi pendidikan tinggi yang cukup baik.

            **Rekomendasi:**  
            Pertahankan kualitas pendidikan dan perkuat keberlanjutan akses ke pendidikan tinggi.
            """)

        elif cluster_id == 1:
            st.markdown("""
            **Karakteristik Klaster 1 (Ketimpangan Menengah)**  
            Didominasi pendidikan menengah dan berada pada fase transisi menuju pendidikan tinggi.

            **Rekomendasi:**  
            Dorong peningkatan akses ke pendidikan tinggi dan penguatan pendidikan vokasi.
            """)

        elif cluster_id == 2:
            st.markdown("""
            **Karakteristik Klaster 2 (Ketimpangan Tinggi)**  
            Proporsi pendidikan rendah masih dominan dan pendidikan tinggi relatif rendah.

            **Rekomendasi:**  
            Prioritaskan penguatan pendidikan dasar dan menengah serta pencegahan putus sekolah.
            """)

    # ==================================================
    # 2. RINGKASAN KARAKTERISTIK KLASTER
    # ==================================================
    st.markdown("---")
    st.subheader("📌 Ringkasan Karakteristik Klaster")

    if len(dfp) >= n_clusters_default:
        cluster_summary = dfp.groupby("cluster")[
            ["rendah_pct", "menengah_pct", "tinggi_pct"]
        ].mean()

        st.dataframe(cluster_summary)

        st.markdown("""
        🟢 **Klaster 0 – Ketimpangan Pendidikan Rendah**

        Klaster ini memiliki proporsi pendidikan rendah dan tinggi yang relatif seimbang, 
        dengan pendidikan menengah sedikit lebih dominan. Struktur ini menunjukkan bahwa 
        kelurahan dalam klaster 0 memiliki distribusi pendidikan yang lebih merata dan 
        akses pendidikan yang relatif lebih baik.

        ➡️ **Makna:**<br/>
        Merepresentasikan kelurahan dengan tingkat ketimpangan pendidikan paling rendah.

        `💡 Rekomendasi:`<br/>
        Pertahankan kualitas pendidikan yang sudah relatif baik dengan fokus pada 
        peningkatan mutu sekolah dan keberlanjutan akses ke pendidikan tinggi.

        ___

        🟡 **Klaster 1 – Ketimpangan Pendidikan Menengah**

        Klaster ini didominasi oleh pendidikan menengah, sementara proporsi pendidikan 
        tinggi relatif rendah. Pendidikan rendah masih cukup signifikan, namun tidak 
        mendominasi sepenuhnya.

        ➡️ **Makna:**<br/>
        Menggambarkan kelurahan yang berada pada fase transisi, di mana sebagian besar 
        penduduk telah mencapai pendidikan menengah tetapi belum banyak yang menempuh 
        pendidikan tinggi.

        `💡 Rekomendasi:`<br/>
        Perlu didorong program lanjutan ke pendidikan tinggi dan vokasi agar lulusan 
        pendidikan menengah tidak berhenti pada jenjang tersebut.

        ___

        🔴 **Klaster 2 – Ketimpangan Pendidikan Tinggi**

        Klaster ini memiliki proporsi pendidikan rendah paling tinggi dan pendidikan tinggi 
        paling rendah dibanding klaster lain. Hal ini menunjukkan keterbatasan akses dan 
        capaian pendidikan lanjutan.

        ➡️ **Makna:**<br/>
        Merepresentasikan kelurahan dengan tingkat ketimpangan pendidikan paling tinggi 
        dan menjadi wilayah prioritas dalam upaya pemerataan pendidikan.

        `💡 Rekomendasi:`<br/>
        Menjadi prioritas utama intervensi, terutama pada penguatan pendidikan dasar, 
        pencegahan putus sekolah, dan peningkatan akses pendidikan menengah.
        """, unsafe_allow_html=True)
    else:
        st.warning("Tidak dapat menampilkan ringkasan klaster karena data terlalu sedikit.")


    # ==================================================
    # 3. VISUALISASI HASIL CLUSTERING
    # ==================================================
    st.markdown("---")
    st.subheader("📊 Visualisasi Hasil Clustering")

    if len(dfp) >= n_clusters_default:
        fig, ax = plt.subplots()
        ax.scatter(
            dfp["rendah_pct"],
            dfp["tinggi_pct"],
            c=dfp["cluster"]
        )
        ax.set_xlabel("Proporsi Pendidikan Rendah")
        ax.set_ylabel("Proporsi Pendidikan Tinggi")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        Visualisasi ini menunjukkan pemisahan kelurahan berdasarkan proporsi
        pendidikan rendah dan tinggi, dengan warna yang merepresentasikan klaster.
        """)
    else:
        st.warning("Tidak dapat menampilkan visualisasi klaster karena data terlalu sedikit.")


    # ==================================================
    # 4. METODOLOGI (SUPPORTING SECTION)
    # ==================================================
    st.markdown("---")
    st.subheader("⚙️ Metodologi K-Means (Elbow Method)")

    if len(dfp) >= 8: # Minimal 8 data untuk range 2-7 cluster
        inertia = []
        K = range(2, 8)

        for k in K:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K, inertia, marker="o")
        ax.set_xlabel("Jumlah Klaster (k)")
        ax.set_ylabel("Inertia")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        Elbow Method digunakan untuk menentukan jumlah klaster optimal.
        Berdasarkan grafik, jumlah klaster **k = 3** dipilih karena memberikan
        keseimbangan antara kompleksitas model dan interpretabilitas hasil.
        """)
    else:
        st.warning("Tidak dapat menampilkan Elbow Method karena data terlalu sedikit.")