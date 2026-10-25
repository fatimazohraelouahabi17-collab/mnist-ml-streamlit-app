import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

from utils.data_loader import get_mnist_sample, get_digit_image

st.set_page_config(page_title="PCA - MNIST", page_icon="📉", layout="wide")
st.title("📉 PCA — Analyse en Composantes Principales")

tab_theorie, tab_code, tab_interactif = st.tabs(["📐 Théorie", "💻 Code", "🎛️ Interactif"])

# ----------------------------------------------------------------------
# ONGLET THÉORIE
# ----------------------------------------------------------------------
with tab_theorie:
    st.header("Intuition")
    st.markdown("""
    MNIST a **784 dimensions** (28×28 pixels). Beaucoup de ces pixels sont
    corrélés entre eux (les coins d'une image sont presque toujours noirs,
    par exemple). La PCA cherche de **nouvelles directions** (des combinaisons
    linéaires des pixels d'origine) qui capturent le maximum de variance
    des données, en éliminant la redondance.
    """)

    st.header("Formulation mathématique")

    st.markdown("**1. Centrage des données**")
    st.latex(r"""
    \tilde{X} = X - \mu, \qquad \mu = \frac{1}{n}\sum_{i=1}^{n} x_i
    """)
    st.caption("On centre chaque pixel autour de sa moyenne sur tout le dataset.")

    st.markdown("**2. Matrice de covariance**")
    st.latex(r"""
    \Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X} \quad \in \mathbb{R}^{784 \times 784}
    """)

    st.markdown("**3. Décomposition en valeurs propres**")
    st.latex(r"""
    \Sigma v_k = \lambda_k v_k
    """)
    st.caption("""
    Chaque vecteur propre vₖ (appelé **composante principale**) donne une direction ;
    la valeur propre associée λₖ mesure la variance des données le long de cette direction.
    On trie les λₖ par ordre décroissant.
    """)

    st.markdown("**4. Projection (réduction de dimension)**")
    st.latex(r"""
    Z = \tilde{X}\, V_k \qquad \text{avec } V_k = [v_1, v_2, \dots, v_k]
    """)
    st.caption("On garde uniquement les k premiers vecteurs propres → réduction de 784 à k dimensions.")

    st.markdown("**5. Variance expliquée**")
    st.latex(r"""
    \text{Variance expliquée par } v_k = \frac{\lambda_k}{\sum_{j=1}^{784} \lambda_j}
    """)

    st.markdown("**6. Reconstruction (approximative)**")
    st.latex(r"""
    \hat{X} = Z V_k^T + \mu
    """)
    st.caption("Permet de mesurer combien d'information a été perdue en compressant les données.")

# ----------------------------------------------------------------------
# ONGLET CODE
# ----------------------------------------------------------------------
with tab_code:
    st.header("Implémentation avec scikit-learn")
    st.code('''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Standardiser les données (moyenne=0, écart-type=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Créer et entraîner la PCA
n_components = 50  # nombre de composantes à garder
pca = PCA(n_components=n_components, random_state=42)
X_reduced = pca.fit_transform(X_scaled)   # shape: (n_samples, n_components)

# 3. Variance expliquée par chaque composante
print(pca.explained_variance_ratio_)        # ex: [0.09, 0.07, 0.06, ...]
print(pca.explained_variance_ratio_.cumsum())  # variance cumulée

# 4. Reconstruction (pour évaluer la perte d'info)
X_reconstructed = pca.inverse_transform(X_reduced)
erreur_reconstruction = ((X_scaled - X_reconstructed) ** 2).mean()
''', language="python")

    st.header("Points clés du code")
    st.markdown("""
    - `pca.components_` contient les vecteurs propres (les `v_k`)
    - `pca.explained_variance_ratio_` correspond aux `λ_k / Σλ_j`
    - `fit_transform` fait centrage + projection en une seule étape
    - Pour visualiser en 2D/3D, on utilise juste `n_components=2` ou `3`
    """)

# ----------------------------------------------------------------------
# ONGLET INTERACTIF
# ----------------------------------------------------------------------
with tab_interactif:
    st.header("Testez la PCA sur MNIST")

    col_params, col_main = st.columns([1, 3])

    with col_params:
        n_samples = st.slider("Nombre d'images (échantillon)", 500, 10000, 3000, step=500)
        n_components_viz = st.radio("Dimensions pour la visualisation", [2, 3], horizontal=True)
        n_components_variance = st.slider("Composantes pour l'analyse de variance", 5, 200, 50, step=5)

    X_train, X_test, y_train, y_test, scaler = get_mnist_sample(n_samples=n_samples)

    with st.spinner("Calcul de la PCA..."):
        pca_viz = PCA(n_components=n_components_viz, random_state=42)
        X_viz = pca_viz.fit_transform(X_train)

        pca_var = PCA(n_components=n_components_variance, random_state=42)
        pca_var.fit(X_train)

    with col_main:
        st.subheader(f"Projection en {n_components_viz}D")
        labels = {"color": "Chiffre"}
        if n_components_viz == 2:
            fig = px.scatter(
                x=X_viz[:, 0], y=X_viz[:, 1],
                color=y_train.astype(str),
                labels={"x": "PC1", "y": "PC2", "color": "Chiffre"},
                title="Projection PCA 2D des chiffres MNIST",
                opacity=0.7,
            )
        else:
            fig = px.scatter_3d(
                x=X_viz[:, 0], y=X_viz[:, 1], z=X_viz[:, 2],
                color=y_train.astype(str),
                labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": "Chiffre"},
                title="Projection PCA 3D des chiffres MNIST",
                opacity=0.7,
            )
        st.plotly_chart(fig, width='stretch')

    st.divider()

    st.subheader("Variance expliquée")
    col1, col2 = st.columns(2)
    with col1:
        cum_var = np.cumsum(pca_var.explained_variance_ratio_)
        fig_var = px.line(
            x=range(1, len(cum_var) + 1), y=cum_var,
            labels={"x": "Nombre de composantes", "y": "Variance expliquée cumulée"},
            title="Variance expliquée cumulée",
        )
        fig_var.add_hline(y=0.95, line_dash="dash", line_color="red",
                           annotation_text="95%")
        st.plotly_chart(fig_var, width='stretch')
    with col2:
        n_for_95 = int(np.argmax(cum_var >= 0.95) + 1)
        st.metric("Variance capturée avec toutes les composantes affichées",
                   f"{cum_var[-1]*100:.1f}%")
        st.metric("Composantes nécessaires pour 95% de variance",
                   f"{n_for_95}" if cum_var[-1] >= 0.95 else f"> {n_components_variance}")
        st.markdown(f"""
        On passe de **784 dimensions** à seulement **{n_for_95 if cum_var[-1] >= 0.95 else n_components_variance}**
        tout en gardant l'essentiel de l'information. C'est l'intérêt principal
        de la PCA : compression avec perte minimale.
        """)

    st.divider()
    st.subheader("Reconstruction d'images")
    st.markdown("Comparaison entre image originale et image reconstruite après compression PCA.")

    idx = st.slider("Choisir une image de l'échantillon", 0, len(X_train) - 1, 0)
    n_comp_reconstruction = st.slider("Composantes utilisées pour la reconstruction", 5, 300, 30, step=5)

    pca_recon = PCA(n_components=n_comp_reconstruction, random_state=42)
    X_recon_space = pca_recon.fit_transform(X_train)
    X_reconstructed = pca_recon.inverse_transform(X_recon_space)

    # dé-standardiser pour un affichage correct si un scaler a été utilisé
    original_img = X_train[idx]
    reconstructed_img = X_reconstructed[idx]

    col_orig, col_recon = st.columns(2)
    with col_orig:
        st.markdown(f"**Original** (chiffre: {y_train[idx]})")
        st.image(get_digit_image(original_img), width=200, clamp=True)
    with col_recon:
        st.markdown(f"**Reconstruit** ({n_comp_reconstruction} composantes)")
        st.image(get_digit_image(reconstructed_img), width=200, clamp=True)
