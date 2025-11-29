import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Music Clustering Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

SELECTED_FEATURES = [
    "danceability", "energy", "acousticness", "instrumentalness",
    "speechiness", "tempo", "valence", "loudness"
]

CLUSTER_DESCRIPTIONS = {
    0: "**High Energy & Danceable** — Party / Upbeat tracks.",
    1: "**Acoustic & Calm** — Relaxing, soothing songs.",
    2: "**Electronic / High Tempo** — EDM / Workout music.",
    3: "**Balanced Modern Pop** — Mainstream chart-style tracks.",
    4: "**Low Energy & Dark Mood** — Emotional / Ambient style songs."
}

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("final_clustered_music_dataset.csv")
        
        required_cols = SELECTED_FEATURES + ["cluster"]
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column: **{col}**")
                return pd.DataFrame()

        return df
    
    except FileNotFoundError:
        st.error("File not found: final_clustered_music_dataset.csv")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

st.title("Music Clustering Explorer")
st.markdown("Analyze cluster profiles, PCA visualization, and audio feature insights.")

st.sidebar.header("Filters")

cluster_list = sorted(df["cluster"].unique())

cluster = st.sidebar.selectbox(
    "Select Cluster",
    cluster_list,
    format_func=lambda x: f"Cluster {x}"
)

show_pca = st.sidebar.checkbox("Show PCA Plot", True)
show_profile = st.sidebar.checkbox("Show Cluster Feature Profile", True)
show_data = st.sidebar.checkbox("Show Songs in Cluster", True)

st.markdown("---")

st.header(f"Cluster {cluster} Overview")
st.markdown(CLUSTER_DESCRIPTIONS.get(cluster, "No description available."))

count = len(df[df["cluster"] == cluster])
st.info(f"Cluster **{cluster}** contains **{count:,}** songs.")

st.markdown("---")

if show_profile:
    st.subheader("Cluster Feature Profile")
    profile = df.groupby("cluster")[SELECTED_FEATURES].mean()
    st.bar_chart(profile.loc[cluster])
    st.markdown("---")

if show_pca:
    st.subheader("PCA Scatter Plot (All Clusters)")

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df[SELECTED_FEATURES])

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=df["cluster"],
        cmap="tab10",
        s=15
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection of Clusters")

    st.pyplot(fig)
    st.markdown("---")

if show_data:
    st.subheader(f"Songs in Cluster {cluster}")
    st.markdown("Showing up to **30 random songs**.")

    descriptive_cols = ["track_name", "artist"]

    display_cols = [col for col in descriptive_cols if col in df.columns]
    display_cols += [col for col in SELECTED_FEATURES if col in df.columns]

    cluster_df = df[df["cluster"] == cluster]

    if len(cluster_df) > 0:
        st.dataframe(cluster_df.sample(min(30, len(cluster_df)))[display_cols])
    else:
        st.warning("No songs found for this cluster.")