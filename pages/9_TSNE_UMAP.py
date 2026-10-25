import streamlit as st
import numpy as np
import plotly.express as px
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

from utils.data_loader import get_mnist_sample

st.set_page_config(page_title="t-SNE & UMAP - MNIST", page_icon="🌌", layout="wide")
st.title("🌌 t-SNE & UMAP — Visualisation Non-Linéaire")

tab_theorie, tab_code, tab_interactif = st.tabs(["📐 Théorie", "💻 Code", "🎛️ Interactif"])

with tab_theorie:
    st.header("Pourquoi pas juste la PCA ?")
    st.markdown("""
    La PCA est **linéaire** : elle ne trouve que des combinaisons linéaires
    des pixels. Or MNIST vit sur une **variété (manifold) non-linéaire** dans
    l'espace à 784 dimensions. t-SNE et UMAP sont conçus pour préserver la
    structure **locale** des données (quels points sont voisins) même quand
    cette structure est très non-linéaire — ils donnent souvent des clusters
    par chiffre beaucoup plus nets que la PCA.
    """)

    st.header("t-SNE (t-distributed Stochastic Neighbor Embedding)")
    st.markdown("**1. Similarités dans l'espace d'origine (haute dimension)**")
    st.latex(r"""
    p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
    """)
    st.caption("σᵢ est ajusté par point pour atteindre une 'perplexité' cible (nombre effectif de voisins).")

    st.markdown("**2. Similarités dans l'espace réduit (basse dimension), avec une distribution t de Student**")
    st.latex(r"""
    q_{ij} = \frac{(1+\|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1+\|y_k - y_l\|^2)^{-1}}
    """)
    st.caption("La queue lourde de la distribution t évite le 'crowding problem' (points entassés au centre).")

    st.markdown("**3. Minimiser la divergence de Kullback-Leibler**")
    st.latex(r"""
    KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
    """)
    st.caption("Optimisé par descente de gradient : on déplace les points y pour que Q ressemble à P.")

    st.header("UMAP (Uniform Manifold Approximation and Projection)")
    st.markdown("""
    Base théorique différente (topologie / géométrie riemannienne), mais
    objectif similaire : construire un graphe de voisinage flou dans l'espace
    d'origine, puis optimiser une projection basse dimension qui a une
    **structure topologique la plus proche possible** (via cross-entropy plutôt que KL).
    """)
    st.latex(r"""
    CE(P, Q) = \sum_{i,j} p_{ij}\log\frac{p_{ij}}{q_{ij}} + (1-p_{ij})\log\frac{1-p_{ij}}{1-q_{ij}}
    """)

    st.header("Comparaison")
    st.markdown("""
    | | PCA | t-SNE | UMAP |
    |---|---|---|---|
    | Type | Linéaire | Non-linéaire | Non-linéaire |
    | Vitesse | Très rapide | Lent | Rapide |
    | Préserve | Variance globale | Structure locale | Structure locale + un peu de globale |
    | Reproductible | Oui | Non (stochastique) | Oui (avec random_state) |
    | Transform sur nouvelles données | Oui | Non (par défaut) | Oui |
    """)

with tab_code:
    st.header("Implémentation")
    st.code('''
from sklearn.manifold import TSNE
import umap

# --- t-SNE ---
tsne = TSNE(
    n_components=2,
    perplexity=30,       # ~nombre de voisins effectifs (typiquement 5-50)
    learning_rate="auto",
    init="pca",           # meilleure init que random
    random_state=42,
)
X_tsne = tsne.fit_transform(X_scaled)   # pas de .transform() séparé : fit_transform seulement

# --- UMAP ---
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,       # taille du voisinage local considéré
    min_dist=0.1,         # distance minimale entre points dans l'espace réduit
    random_state=42,
)
X_umap = reducer.fit_transform(X_scaled)
# UMAP supporte .transform() sur de nouvelles données, contrairement à t-SNE
''', language="python")
    st.markdown("""
    **Points clés:**
    - t-SNE : toujours utiliser `init="pca"` et `learning_rate="auto"` (recommandation officielle scikit-learn)
    - `perplexity` (t-SNE) et `n_neighbors` (UMAP) contrôlent le compromis structure locale / globale
    - UMAP est généralement **10-20x plus rapide** que t-SNE sur de gros datasets
    """)

with tab_interactif:
    st.header("Comparer PCA, t-SNE et UMAP côte à côte")

    col_params, _ = st.columns([1, 2])
    with col_params:
        n_samples = st.slider("Nombre d'images", 500, 5000, 1500, step=250, key="tsne_n",
                                help="t-SNE devient lent au-delà de 3000-4000 points")
        perplexity = st.slider("Perplexité (t-SNE)", 5, 50, 30)
        n_neighbors = st.slider("n_neighbors (UMAP)", 5, 50, 15)
        min_dist = st.slider("min_dist (UMAP)", 0.0, 0.99, 0.1)

    X_train, X_test, y_train, y_test, scaler = get_mnist_sample(n_samples=n_samples)
    labels_str = y_train.astype(str)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("PCA")
        t0 = time.time()
        X_pca = PCA(n_components=2, random_state=42).fit_transform(X_train)
        t_pca = time.time() - t0
        fig_pca = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=labels_str,
                              labels={"color": "Chiffre"}, opacity=0.7)
        fig_pca.update_layout(height=450)
        st.plotly_chart(fig_pca, width='stretch')
        st.caption(f"⏱️ {t_pca:.2f}s")

    with col2:
        st.subheader("t-SNE")
        with st.spinner("Calcul t-SNE (le plus lent des 3)..."):
            t0 = time.time()
            X_tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                           learning_rate="auto", random_state=42).fit_transform(X_train)
            t_tsne = time.time() - t0
        fig_tsne = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=labels_str,
                               labels={"color": "Chiffre"}, opacity=0.7)
        fig_tsne.update_layout(height=450)
        st.plotly_chart(fig_tsne, width='stretch')
        st.caption(f"⏱️ {t_tsne:.2f}s")

    with col3:
        st.subheader("UMAP")
        with st.spinner("Calcul UMAP..."):
            t0 = time.time()
            X_umap = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                 min_dist=min_dist, random_state=42).fit_transform(X_train)
            t_umap = time.time() - t0
        fig_umap = px.scatter(x=X_umap[:, 0], y=X_umap[:, 1], color=labels_str,
                               labels={"color": "Chiffre"}, opacity=0.7)
        fig_umap.update_layout(height=450)
        st.plotly_chart(fig_umap, width='stretch')
        st.caption(f"⏱️ {t_umap:.2f}s")

    st.divider()
    st.info(f"""
    **Comparaison des temps de calcul** — PCA: {t_pca:.2f}s | t-SNE: {t_tsne:.2f}s | UMAP: {t_umap:.2f}s

    Généralement, tu devrais voir des clusters de chiffres **beaucoup plus nets et séparés**
    avec t-SNE et UMAP qu'avec la PCA, mais PCA reste largement la plus rapide.
    """)
