import streamlit as st
import numpy as np
import plotly.express as px
import warnings
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.data_loader import get_mnist_sample

st.set_page_config(page_title="QDA - MNIST", page_icon="🌀", layout="wide")
st.title("🌀 QDA — Analyse Discriminante Quadratique")

tab_theorie, tab_code, tab_interactif = st.tabs(["📐 Théorie", "💻 Code", "🎛️ Interactif"])

with tab_theorie:
    st.header("Différence avec LDA")
    st.markdown("""
    LDA suppose que **toutes les classes partagent la même matrice de covariance** Σ.
    QDA relâche cette hypothèse : **chaque classe a sa propre matrice** Σ_c.
    Résultat : les frontières de décision deviennent des **courbes quadratiques**
    (ellipses, paraboles) au lieu de droites.
    """)

    st.header("Formulation mathématique")
    st.markdown("**1. Covariance propre à chaque classe**")
    st.latex(r"""
    \Sigma_c = \frac{1}{n_c - 1}\sum_{i \in c} (x_i - \mu_c)(x_i - \mu_c)^T
    """)

    st.markdown("**2. Règle de décision (fonction discriminante quadratique)**")
    st.latex(r"""
    \delta_c(x) = -\frac{1}{2}\log|\Sigma_c| - \frac{1}{2}(x-\mu_c)^T \Sigma_c^{-1} (x-\mu_c) + \log \pi_c
    """)
    st.latex(r"""
    \hat{y} = \arg\max_c \; \delta_c(x)
    """)

    st.header("⚠️ Le problème : covariance singulière")
    st.markdown("""
    MNIST a **784 dimensions**. Pour que Σ_c soit inversible, il faut au moins
    **784 échantillons indépendants par classe**. Avec un petit échantillon
    (ex: 3000 images / 10 classes ≈ 300 par classe), Σ_c devient **singulière**
    (non-inversible) → `Σ_c⁻¹` n'existe pas → erreurs numériques ou résultats aberrants.
    """)

    st.header("Solutions")
    st.markdown("""
    1. **Réduire la dimension avant QDA** (PCA à ~30-50 composantes) — solution utilisée ici par défaut
    2. **Régularisation** via `reg_param` : mélange Σ_c avec une version "shrinkée"
    """)
    st.latex(r"""
    \Sigma_c^{reg} = (1-\alpha)\,\Sigma_c + \alpha \cdot \frac{\text{tr}(\Sigma_c)}{d}\, I_d
    """)
    st.caption("α = reg_param. Plus α est grand, plus on force la matrice vers une forme diagonale (toujours inversible).")

with tab_code:
    st.header("Implémentation")
    st.code('''
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

# --- Approche 1 : réduire la dimension d'abord (recommandé) ---
pca = PCA(n_components=40, random_state=42)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_reduced, y_train)
y_pred = qda.predict(X_test_reduced)

# --- Approche 2 : régularisation directe sur les 784 dimensions ---
qda_reg = QuadraticDiscriminantAnalysis(reg_param=0.3)  # entre 0 et 1
qda_reg.fit(X_train_scaled, y_train)   # peut quand même être instable si reg_param trop faible
''', language="python")

    st.markdown("""
    **Points clés:**
    - `reg_param=0` (défaut) = QDA classique, risque d'erreur en haute dimension
    - `reg_param=1` = équivalent à une covariance diagonale (très stable, mais perd l'info de corrélation entre pixels)
    - La combinaison **PCA + QDA** est la pratique standard pour ce genre de cas
    """)

with tab_interactif:
    st.header("Comparaison : QDA avec et sans réduction de dimension")

    col_params, _ = st.columns([1, 2])
    with col_params:
        n_samples = st.slider("Nombre d'images", 1000, 10000, 3000, step=500, key="qda_n")
        n_components = st.slider("Composantes PCA (pour l'approche recommandée)", 5, 100, 40, step=5)
        reg_param = st.slider("reg_param (pour l'approche sans PCA)", 0.0, 1.0, 0.0, step=0.05)

    X_train, X_test, y_train, y_test, scaler = get_mnist_sample(n_samples=n_samples)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Avec PCA d'abord")
        pca = PCA(n_components=n_components, random_state=42)
        X_train_r = pca.fit_transform(X_train)
        X_test_r = pca.transform(X_test)
        qda1 = QuadraticDiscriminantAnalysis()
        qda1.fit(X_train_r, y_train)
        acc1 = accuracy_score(y_test, qda1.predict(X_test_r))
        st.metric("Accuracy", f"{acc1*100:.2f}%")
        st.success(f"Stable : {n_components} dimensions << {int(0.9*len(X_train)/10)} échantillons/classe environ")

    with col2:
        st.subheader("⚠️ Sans réduction (784 dims)")
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                qda2 = QuadraticDiscriminantAnalysis(reg_param=reg_param)
                qda2.fit(X_train, y_train)
                acc2 = accuracy_score(y_test, qda2.predict(X_test))
                if len(w) > 0:
                    st.warning(f"⚠️ {len(w)} warning(s) détecté(s) — ex: '{str(w[0].message)[:100]}...'")
            st.metric("Accuracy", f"{acc2*100:.2f}%")
            if reg_param == 0:
                st.error("reg_param=0 → risque élevé d'instabilité numérique en 784 dimensions")
            else:
                st.info(f"reg_param={reg_param} régularise la covariance et stabilise les calculs")
        except np.linalg.LinAlgError as e:
            st.error(
                "❌ **Erreur fatale : matrice de covariance singulière !**\n\n"
                f"`{str(e)[:200]}`\n\n"
                "C'est exactement le problème théorique expliqué dans l'onglet "
                "'Théorie' : trop peu d'échantillons par classe par rapport aux "
                "784 dimensions. **Augmente `reg_param`** (essaie 0.3 par exemple) "
                "pour régulariser la covariance et éviter cette erreur."
            )
            acc2 = None

    st.divider()
    st.subheader("Matrice de confusion (approche recommandée : PCA + QDA)")
    cm = confusion_matrix(y_test, qda1.predict(X_test_r))
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Purples",
        labels=dict(x="Prédiction", y="Vraie classe", color="Nombre"),
        x=[str(i) for i in range(10)], y=[str(i) for i in range(10)],
    )
    st.plotly_chart(fig_cm, width='stretch')
