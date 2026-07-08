import streamlit as st

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Multivariate Analysis Tool", page_icon="📊", layout="wide")

# ---- HOMEPAGE CONTENT ----
st.title("📊 Multivariate Analysis Tool")
st.markdown("Explore complex data relationships using **11 powerful statistical methods**.")

# ---- FEATURE CARDS (Row 1) ----
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔍 1. Correlation")
    st.write("Pearson, Spearman, Kendall with heatmaps.")
    if st.button("Go to Correlation", use_container_width=True):
        st.switch_page("pages/01_Correlation.py")

with col2:
    st.subheader("📈 2. Regression")
    st.write("Linear, Polynomial, Ridge, Lasso.")
    if st.button("Go to Regression", use_container_width=True):
        st.switch_page("pages/02_Regression.py")

with col3:
    st.subheader("🌲 3. Hierarchical Clustering")
    st.write("Dendrograms, silhouette scores.")
    if st.button("Go to Clustering", use_container_width=True):
        st.switch_page("pages/03_Hierarchical_Clustering.py")

# ---- FEATURE CARDS (Row 2) ----
col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("🎯 4. K-Means")
    st.write("Elbow plots, silhouette analysis.")
    if st.button("Go to K-Means", use_container_width=True):
        st.switch_page("pages/04_KMeans.py")

with col5:
    st.subheader("📊 5. LDA / QDA")
    st.write("Linear & Quadratic Discriminant Analysis.")
    if st.button("Go to LDA/QDA", use_container_width=True):
        st.switch_page("pages/05_LDA_QDA.py")

with col6:
    st.subheader("🧩 6. Factor Analysis")
    st.write("Exploratory Factor Analysis (EFA).")
    if st.button("Go to Factor Analysis", use_container_width=True):
        st.switch_page("pages/06_Factor_Analysis.py")

# ---- FEATURE CARDS (Row 3) ----
col7, col8, col9 = st.columns(3)

with col7:
    st.subheader("📉 7. MANOVA")
    st.write("Multivariate ANOVA with Wilks' lambda.")
    if st.button("Go to MANOVA", use_container_width=True):
        st.switch_page("pages/07_MANOVA.py")

with col8:
    st.subheader("🧬 8. Logistic Regression")
    st.write("Binary outcome, ROC curves.")
    if st.button("Go to Logistic", use_container_width=True):
        st.switch_page("pages/08_Logistic_Regression.py")

with col9:
    st.subheader("🔗 9. Canonical Correlation")
    st.write("CCA – explore relationships.")
    if st.button("Go to CCA", use_container_width=True):
        st.switch_page("pages/09_CCA.py")

# ---- FEATURE CARDS (Row 4) ----
col10, col11 = st.columns(2)

with col10:
    st.subheader("🕸️ 10. t-SNE / UMAP")
    st.write("Dimensionality reduction.")
    if st.button("Go to t-SNE/UMAP", use_container_width=True):
        st.switch_page("pages/10_tsne_umap.py")

with col11:
    st.subheader("⚙️ 11. SEM")
    st.write("Structural Equation Modeling.")
    if st.button("Go to SEM", use_container_width=True):
        st.switch_page("pages/11_SEM.py")

# ---- SIDEBAR ----
st.sidebar.success("Select a method from the sidebar or click a card above.")
st.sidebar.info("Upload your data on each analysis page.")
