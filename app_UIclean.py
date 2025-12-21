import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Dashboard Ketimpangan Pendidikan Kabupaten Bandung",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    /* Container tab */
    div[data-baseweb="tab-list"] {
        justify-content: center;
        gap: 30px;
        margin-bottom: 25px;
    }

    /* Tab item */
    button[data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 600;
        padding: 12px 28px;
        border-radius: 10px;
    }

    /* Tab aktif */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #f0f2f6;
        border-bottom: 3px solid #ff7e5f;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================
# GLOBAL HEADER
# ======================================================
st.markdown(
    """
    <div style="
        background-color:#f5f7fa;
        padding:25px;
        border-radius:12px;
        text-align:center;
        margin-bottom:25px;
    ">
        <h1 style="margin-bottom:5px;">🎓 Ketimpangan Pendidikan Antar Kelurahan</h1>
        <p style="margin-top:0; color:#555;">
            Gambaran kondisi pendidikan di Kabupaten Bandung
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset_final.csv", index_col=0)
    except FileNotFoundError:
        st.error("File dataset_final.csv tidak ditemukan.")
        df = pd.DataFrame({
            "rendah_pct": [0.3, 0.6, 0.4],
            "menengah_pct": [0.4, 0.3, 0.4],
            "tinggi_pct": [0.3, 0.1, 0.2]
        }, index=["Kelurahan A", "Kelurahan B", "Kelurahan C"])
    return df

dfp = load_data()

# ======================================================
# NAVIGATION
# ======================================================
menu = st.tabs([
    "🏠  Ringkasan",
    "🔎  Eksplorasi Data",
    "🎯  Clustering"
])

# ======================================================
# 🏠 RINGKASAN
# ======================================================
with menu[0]:
    st.subheader("🎯 Gambaran Singkat")

    st.markdown("""
    - Kondisi pendidikan **belum merata** antar kelurahan di Kabupaten Bandung  
    - Kelurahan dengan banyak warga berpendidikan rendah **cenderung memiliki lebih sedikit warga berpendidikan tinggi**  
    - Berdasarkan pola tersebut, kelurahan dapat dikelompokkan menjadi **3 kelompok utama**
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("📍 Kelurahan Dianalisis", len(dfp))
    col2.metric("📊 Indikator Pendidikan", 3)
    col3.metric("🎯 Jumlah Kelompok", 3)

    st.info(
        "Dashboard ini memberikan gambaran awal untuk membantu "
        "menentukan wilayah yang perlu diprioritaskan dalam upaya pemerataan pendidikan."
    )

# ======================================================
# 🔎 EKSPLORASI DATA
# ======================================================
with menu[1]:
    st.subheader("Gambaran Pola Pendidikan Antar Kelurahan")

    st.markdown("""
    **Apa yang ingin dilihat di bagian ini?**  
    Bagian ini menampilkan pola umum pendidikan antar kelurahan untuk menunjukkan
    bahwa kondisi pendidikan di Kabupaten Bandung tidak seragam.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Scatter plot
    with col1:
        st.markdown("**Perbandingan Pendidikan Rendah dan Tinggi**")
        st.caption(
            "Setiap titik mewakili satu kelurahan."
        )

        fig, ax = plt.subplots()
        ax.scatter(dfp["rendah_pct"], dfp["tinggi_pct"])
        ax.set_xlabel("Persentase Pendidikan Rendah")
        ax.set_ylabel("Persentase Pendidikan Tinggi")
        ax.set_title("Pendidikan Rendah vs Pendidikan Tinggi")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Kelurahan dengan pendidikan rendah yang tinggi umumnya memiliki "
            "pendidikan tinggi yang lebih sedikit."
        )

    # Heatmap korelasi
    with col2:
        st.markdown("**Keterkaitan Antar Tingkat Pendidikan**")
        st.caption(
            "Diagram ini membantu melihat hubungan antar indikator pendidikan."
        )

        fig, ax = plt.subplots()
        sns.heatmap(
            dfp[["rendah_pct", "menengah_pct", "tinggi_pct"]].corr(),
            annot=True,
            cmap="Blues",
            ax=ax
        )
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Jika satu tingkat pendidikan mendominasi, "
            "tingkat pendidikan lainnya cenderung lebih rendah."
        )

    st.info(
        "Perbedaan pola ini menjadi dasar perlunya pengelompokan kelurahan "
        "agar kebijakan pendidikan dapat disesuaikan dengan kondisi wilayah."
    )

# ======================================================
# 🎯 CLUSTERING
# ======================================================
with menu[2]:
    st.subheader("Pengelompokan Kelurahan Berdasarkan Pendidikan")

    # Layout 2 kolom
    col_setting, col_result = st.columns([1, 2])

    # --------------------------
    # LEFT PANEL – Pengaturan
    # --------------------------
    with col_setting:

        st.markdown("### Pengaturan Analisis")

        k = st.radio(
            "Jumlah Kelompok Kelurahan",
            options=[2, 3],
            index=1,
            format_func=lambda x: f"{x} Kelompok"
        )

        st.caption(
            "Jumlah kelompok menentukan pemisahan karakteristik ketimpangan "
            "pendidikan antar kelurahan."
        )

        st.markdown("### Indikator yang Digunakan")
        st.success("✔ Pendidikan Rendah (%)")
        st.success("✔ Pendidikan Menengah (%)")
        st.success("✔ Pendidikan Tinggi (%)")

        st.markdown("### Metode")
        st.radio("", ["K-Means"], index=0, disabled=True)

    # --------------------------
    # PROSES CLUSTERING
    # --------------------------
    features = ["rendah_pct", "menengah_pct", "tinggi_pct"]
    X = dfp[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    dfp["cluster"] = kmeans.fit_predict(X_scaled)

    # Hitung persentase setiap cluster
    cluster_counts = dfp["cluster"].value_counts(normalize=True) * 100
    cluster_counts = cluster_counts.round(1)

    # --------------------------
    # LEFT PANEL – tampilkan dataset
    # --------------------------
    with col_setting:
        st.markdown("---")
        st.markdown("### 📂 Dataset Setelah Clustering")

        st.dataframe(
            dfp.style.format(
                "{:.2f}",
                subset=["rendah_pct","menengah_pct","tinggi_pct"]
            ),
            height=350
        )

    # --------------------------
    # RIGHT PANEL – visual & info
    # --------------------------
    with col_result:

        st.markdown("### 📊 Visualisasi Hasil Clustering")

        cluster_colors = {0:"#2ecc71", 1:"#f1c40f", 2:"#e74c3c"}
        colors = dfp["cluster"].map(cluster_colors)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            dfp["rendah_pct"],
            dfp["tinggi_pct"],
            c=colors,
            s=60,
            alpha=0.85
        )

        ax.set_xlabel("Pendidikan Rendah (%)")
        ax.set_ylabel("Pendidikan Tinggi (%)")
        ax.set_title("Pola Ketimpangan Pendidikan Antar Kelurahan")

        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                label=f'Ketimpangan Rendah ({cluster_counts.get(0,0)}%)',
                markerfacecolor="#2ecc71"),
            Line2D([0], [0], marker='o', color='w',
                label=f'Ketimpangan Sedang ({cluster_counts.get(1,0)}%)',
                markerfacecolor="#f1c40f"),
            Line2D([0], [0], marker='o', color='w',
                label=f'Ketimpangan Tinggi ({cluster_counts.get(2,0)}%)',
                markerfacecolor="#e74c3c"),
        ]
        ax.legend(handles=legend_elements, title="Makna Kelompok")

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Ringkasan Makna Kelompok")
        st.markdown("""
        🟢 **Ketimpangan Rendah** – Struktur pendidikan relatif seimbang  
        🟡 **Ketimpangan Sedang** – Didominasi pendidikan menengah  
        🔴 **Ketimpangan Tinggi** – Pendidikan rendah masih mendominasi
        """)

        st.subheader("🔍 Cek Kelompok Kelurahan")
        kelurahan = st.selectbox("Pilih Kelurahan", dfp.index.tolist())
        cid = dfp.loc[kelurahan, "cluster"]

        st.write(
            f"Kelurahan **{kelurahan}** termasuk dalam **Kelompok {cid + 1}**"
        )

        if cid == 0:
            st.success("Wilayah dengan ketimpangan pendidikan relatif rendah.")
        elif cid == 1:
            st.warning("Wilayah dengan ketimpangan pendidikan sedang.")
        else:
            st.error("Wilayah dengan ketimpangan pendidikan tinggi dan perlu perhatian khusus.")
