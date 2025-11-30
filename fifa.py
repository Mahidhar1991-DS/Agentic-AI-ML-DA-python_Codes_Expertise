# fifa_advanced_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from ydata_profiling import ProfileReport
import tempfile
import io

sns.set_style("whitegrid")
st.set_page_config("FIFA Advanced EDA & Insights", layout="wide")

# -------------------------
# Helpers & cleaning utils
# -------------------------
@st.cache_data(show_spinner=False)
def convert_money_col(series: pd.Series) -> pd.Series:
    def convert_money(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return x
        s = str(x).replace("€", "").replace(",", "").strip()
        try:
            if s.endswith("M") or "M" in s:
                return float(s.replace("M", "")) * 1_000_000
            if s.endswith("K") or "K" in s:
                return float(s.replace("K", "")) * 1_000
            return float(s)
        except:
            return np.nan
    return series.map(convert_money)

@st.cache_data(show_spinner=False)
def parse_height(h):
    # Heights like "5'7", "6'2" or "170cm"
    try:
        if pd.isna(h): return np.nan
        s = str(h).strip()
        if "cm" in s:
            return float(s.replace("cm","").strip())
        if "'" in s:
            parts = s.split("'")
            feet = float(parts[0])
            inches = float(parts[1]) if parts[1] else 0.0
            return round((feet*12 + inches) * 2.54, 1)
        # fallback numeric
        return float(s)
    except:
        return np.nan

@st.cache_data(show_spinner=False)
def parse_weight(w):
    # Weight like "75kg" or "165lbs"
    try:
        if pd.isna(w): return np.nan
        s = str(w).strip().lower().replace(" ", "")
        if "kg" in s:
            return float(s.replace("kg",""))
        if "lbs" in s or "lb" in s:
            val = float(s.replace("lbs","").replace("lb",""))
            return round(val * 0.453592, 1)
        return float(s)
    except:
        return np.nan

@st.cache_data(show_spinner=False)
def clean_fifa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # lowercase column names for convenience
    df.columns = [c.strip() for c in df.columns]

    # Convert money columns if present
    for c in ["Value", "Wage", "Release Clause", "Value "]:
        if c in df.columns:
            df[c] = convert_money_col(df[c])

    # Height & Weight
    if "Height" in df.columns:
        df["Height_cm"] = df["Height"].apply(parse_height)
    if "Weight" in df.columns:
        df["Weight_kg"] = df["Weight"].apply(parse_weight)

    # Fill some common missing columns or convert types
    if "Overall" in df.columns:
        df["Overall"] = pd.to_numeric(df["Overall"], errors="coerce")
    if "Potential" in df.columns:
        df["Potential"] = pd.to_numeric(df["Potential"], errors="coerce")
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Simplify club and nationality missing
    if "Club" in df.columns:
        df["Club"] = df["Club"].fillna("No Club")
    if "Nationality" in df.columns:
        df["Nationality"] = df["Nationality"].fillna("Unknown")

    return df

# -------------------------
# UI: Sidebar - Upload & Options
# -------------------------
st.sidebar.title("FIFA — Advanced Streamlit")
st.sidebar.markdown("Upload your raw FIFA CSV (like data.csv) and explore.")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

generate_profile = st.sidebar.checkbox("Enable full automated Profile report (may be slow)", value=False)

# Quick sample dataset fallback (optional)
use_sample = st.sidebar.checkbox("Use sample fifa (if no file)", value=False)

# -------------------------
# Load data
# -------------------------
@st.cache_data(show_spinner=True)
def load_sample():
    # small synthetic sample or fallback - here we attempt to load uploaded path if exists
    # In your environment, you can replace with local file path.
    return pd.DataFrame()

df = None
if uploaded:
    with st.spinner("Loading dataset..."):
        df = pd.read_csv(r'/Users/mahidharreddy/Downloads/Data science/Nov/26-27- Nov/25th, 26th- Advanced EDA project/Seaborn - SPORT/FIFA.csv')
elif use_sample:
    st.info("No file uploaded: using small sample extracted from your earlier data.")
    # create extremely small sample with expected columns if file not provided
    df = pd.DataFrame({
        "Name": ["L. Messi","C. Ronaldo","Neymar Jr","A Youngster"],
        "Age":[31,33,26,18],
        "Nationality":["Argentina","Portugal","Brazil","England"],
        "Club":["FC Barcelona","Juventus","Paris Saint-Germain","Youth FC"],
        "Overall":[94,94,92,65],
        "Potential":[94,94,93,80],
        "Value":["€226.5M","€127.1M","€228.1M","€300K"],
        "Wage":["€565K","€500K","€400K","€1K"],
        "Preferred Foot":["Left","Right","Right","Right"],
        "Height":["170cm","187cm","175cm","180cm"],
        "Weight":["72kg","83kg","68kg","70kg"]
    })
else:
    st.info("Upload a FIFA CSV in the sidebar to start (or tick 'Use sample').")

if df is None:
    st.stop()

# -------------------------
# Clean data (cached)
# -------------------------
with st.spinner("Cleaning dataset..."):
    df_clean = clean_fifa(df)

# -------------------------
# Top-level metrics
# -------------------------
st.header("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", df_clean.shape[0])
col2.metric("Cols", df_clean.shape[1])
col3.metric("Missing cells", int(df_clean.isna().sum().sum()))
col4.metric("Unique Nationalities", df_clean["Nationality"].nunique() if "Nationality" in df_clean.columns else 0)

# -------------------------
# Tabs for sections
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Interactive EDA", "Player Explorer", "Similarity & Clustering", "Export / Report", "Notes"])

# -------------------------
# Tab 1: Interactive EDA
# -------------------------
with tab1:
    st.subheader("Univariate / Bivariate Visuals")
    st.write("Use controls to build plots. Supports seaborn and Plotly interactive charts.")

    colA, colB = st.columns([2,1])
    with colB:
        plot_lib = st.radio("Plot type", options=["Seaborn (static)", "Plotly (interactive)"])
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
        selected_numeric = st.multiselect("Numeric columns (for pair/corr)", numeric_cols, default=["Overall","Potential","Age"][:len(numeric_cols)])
        corr_method = st.selectbox("Correlation method", ["pearson","spearman","kendall"])
    with colA:
        st.markdown("### Quick Correlation")
        if len(selected_numeric) >= 2:
            corr = df_clean[selected_numeric].corr(method=corr_method)
            fig_corr, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig_corr)
        else:
            st.info("Pick at least 2 numeric columns to show correlation.")

    # scatter with hue
    st.markdown("### Scatter (X vs Y)")
    x = st.selectbox("X", numeric_cols, index= numeric_cols.index("Overall") if "Overall" in numeric_cols else 0)
    y = st.selectbox("Y", numeric_cols, index= numeric_cols.index("Potential") if "Potential" in numeric_cols else (1 if len(numeric_cols)>1 else 0))
    hue = st.selectbox("Color by (categorical)", [None] + cat_cols)
    if plot_lib == "Seaborn (static)":
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(data=df_clean, x=x, y=y, hue=hue, alpha=0.7, ax=ax)
        st.pyplot(fig)
    else:
        fig = px.scatter(df_clean, x=x, y=y, color=hue, hover_data=["Name"] if "Name" in df_clean.columns else None)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Distribution")
    dist_col = st.selectbox("Select numeric for distribution", numeric_cols, index=0)
    if plot_lib == "Seaborn (static)":
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df_clean[dist_col].dropna(), kde=True, bins=30, ax=ax)
        st.pyplot(fig)
    else:
        fig = px.histogram(df_clean, x=dist_col, nbins=30)
        st.plotly_chart(fig)

# -------------------------
# Tab 2: Player Explorer
# -------------------------
with tab2:
    st.subheader("Player Explorer & Filters")
    cols = st.columns(4)
    name_search = st.text_input("Search Name (partial)", "")
    nat_filter = st.multiselect("Nationality", options=sorted(df_clean["Nationality"].unique()) if "Nationality" in df_clean.columns else [])
    club_filter = st.multiselect("Club", options=sorted(df_clean["Club"].unique()) if "Club" in df_clean.columns else [])

    df_view = df_clean.copy()
    if name_search:
        df_view = df_view[df_view["Name"].str.contains(name_search, case=False, na=False)]
    if nat_filter:
        df_view = df_view[df_view["Nationality"].isin(nat_filter)]
    if club_filter:
        df_view = df_view[df_view["Club"].isin(club_filter)]

    st.dataframe(df_view.reset_index(drop=True).head(200))

    if st.button("Show top 10 by Overall"):
        st.table(df_clean.sort_values("Overall", ascending=False)[["Name","Club","Nationality","Overall","Potential","Value","Wage"]].head(10).reset_index(drop=True))

# -------------------------
# Tab 3: Similarity & Clustering
# -------------------------
with tab3:
    st.subheader("Player similarity (PCA + Cosine) and Clustering (KMeans)")

    # features used for similarity
    features = st.multiselect("Select numeric features for similarity/clustering",
                              options=[c for c in df_clean.select_dtypes(include=[np.number]).columns.tolist() if df_clean[c].nunique()>5],
                              default=["Overall","Potential","Age","Stamina"] if set(["Overall","Potential","Age","Stamina"]).issubset(df_clean.columns) else df_clean.select_dtypes(include=[np.number]).columns.tolist()[:4])

    if len(features) < 2:
        st.info("Pick at least 2 numeric features.")
    else:
        X = df_clean[features].fillna(df_clean[features].median())
        # PCA reduce to 10 dims max for speed
        pca_n = min(10, X.shape[1])
        pca = PCA(n_components=min(6, pca_n))
        X_pca = pca.fit_transform(X)

        # KMeans clustering
        n_clusters = st.slider("KMeans clusters", min_value=2, max_value=12, value=4)
        km = KMeans(n_clusters=n_clusters, random_state=42)
        labels = km.fit_predict(X_pca)
        df_clean["_cluster"] = labels

        # Scatter plot of first two PCA components
        fig = px.scatter(pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]).assign(Name=df_clean["Name"]),
                         x="PC1", y="PC2", color=labels.astype(str), hover_name=df_clean["Name"])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Show cluster centroids (in original feature space)")
        centroids = pd.DataFrame(km.cluster_centers_, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
        st.dataframe(centroids)

        # Similarity: pick a player
        st.markdown("### Find most similar players")
        player = st.selectbox("Pick a player", df_clean["Name"].tolist())
        top_n = st.slider("Top N similar players", 1, 20, 5)
        idx = df_clean[df_clean["Name"] == player].index[0]
        sims = cosine_similarity(X.fillna(0))
        sim_scores = sims[idx]
        sim_idx = np.argsort(-sim_scores)[1:top_n+1]
        sim_df = df_clean.iloc[sim_idx][["Name","Club","Nationality"] + features].copy()
        sim_df["score"] = sim_scores[sim_idx]
        st.table(sim_df.reset_index(drop=True))

# -------------------------
# Tab 4: Export / Report
# -------------------------
with tab4:
    st.subheader("Export cleaned data & generate automated report")

    # Download cleaned CSV
    csv = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned CSV", data=csv, file_name="fifa_cleaned.csv", mime="text/csv")

    # Generate & display full profile report (ydata-profiling) if user allowed
    if generate_profile:
        if st.button("Generate Profile Report (HTML)"):
            with st.spinner("Generating profile report (this can take time)..."):
                profile = ProfileReport(df_clean, title="FIFA Profile", explorative=True)
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                    profile.to_file(tmp.name)
                    html_bytes = open(tmp.name, "rb").read()
                    st.download_button("Download Profile HTML", data=html_bytes, file_name="fifa_profile.html", mime="text/html")
                    st.success("Profile generated — you can download the HTML report.")

# -------------------------
# Tab 5: Notes & next steps
# -------------------------
with tab5:
    st.header("Notes & Next Steps")
    st.markdown("""
    - This app expects the raw FIFA CSV. It cleans money, height, weight and common numeric columns.
    - For very large files (>100k rows) avoid generating the full profile inside Streamlit (use offline).
    - Next features you can add:
        * Time-series if you have historical seasons,
        * Advanced ML prediction (value/overall),
        * Auth & role-based access for clients,
        * Database backend (Postgres) and scheduled ingestion,
        * Hosting: Streamlit Cloud / Heroku / AWS Elastic Beanstalk / Docker.
    """)
    st.markdown("Created by your assistant — ask me to add a new feature (dark mode, PDF export, ML model, UI polish).")
