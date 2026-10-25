import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils.data_loader import get_mnist_sample

st.set_page_config(page_title="LDA - MNIST", page_icon="📊", layout="wide")
st.title("📊 LDA — Analyse Discriminante Linéaire")

tab_theorie, tab_code, tab_interactif = st.tabs(["📐 Théorie", "💻 Code", "🎛️ Interactif"])

# ----------------------------------------------------------------------
# ONGLET THÉORIE
# ----------------------------------------------------------------------
with tab_theorie:
    st.header("Intuition")
    st.markdown("""
    Contrairement à la PCA (non-supervisée, cherche la variance max), la **LDA
    est supervisée** : elle utilise les labels (les vrais chiffres 0-9) pour
    trouver les directions qui **séparent le mieux les classes**.

    Objectif : maximiser la distance **entre** les centres des classes,
    tout en minimisant la dispersion **à l'intérieur** de chaque classe.
    """)

    st.header("Formulation mathématique")

    st.markdown("**1. Moyennes de classes et moyenne globale**")
    st.latex(r"""
    \mu_c = \frac{1}{n_c}\sum_{i \in c} x_i \qquad
    \mu = \frac{1}{n}\sum_{i=1}^{n} x_i
    """)

    st.markdown("**2. Matrice de dispersion intra-classe (within-class)**")
    st.latex(r"""
    S_W = \sum_{c=1}^{C} \sum_{i \in c} (x_i - \mu_c)(x_i - \mu_c)^T
    """)
    st.caption("Mesure à quel point les points d'une même classe sont dispersés autour de leur centre.")

    st.markdown("**3. Matrice de dispersion inter-classe (between-class)**")
    st.latex(r"""
    S_B = \sum_{c=1}^{C} n_c (\mu_c - \mu)(\mu_c - \mu)^T
    """)
    st.caption("Mesure à quel point les centres des différentes classes sont éloignés les uns des autres.")

    st.markdown("**4. Critère de Fisher à maximiser**")
    st.latex(r"""
    J(w) = \frac{w^T S_B w}{w^T S_W w}
    """)

    st.markdown("**5. Solution : problème aux valeurs propres généralisées**")
    st.latex(r"""
    S_W^{-1} S_B\, w_k = \lambda_k w_k
    """)
    st.caption("""
    Les vecteurs propres wₖ (triés par λₖ décroissant) donnent les directions
    discriminantes. Contrairement à la PCA, le nombre maximal de composantes
    utiles est **C - 1** (C = nombre de classes), soit **9 pour MNIST** (10 chiffres).
    """)

    st.markdown("**6. Classification (règle de décision)**")
    st.latex(r"""
    \hat{y} = \arg\max_{c} \; \left( x^T \Sigma^{-1} \mu_c - \frac{1}{2}\mu_c^T \Sigma^{-1} \mu_c + \log \pi_c \right)
    """)
    st.caption("""
    LDA suppose que toutes les classes partagent la **même** matrice de
    covariance Σ → les frontières de décision entre classes sont des
    **hyperplans linéaires** (d'où le nom).
    """)

# ----------------------------------------------------------------------
# ONGLET CODE
# ----------------------------------------------------------------------
with tab_code:
    st.header("Implémentation avec scikit-learn")
    st.code('''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# 1. Standardiser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Entraîner la LDA (utilisée ici comme CLASSIFIEUR)
lda = LinearDiscriminantAnalysis(solver="svd")
lda.fit(X_train_scaled, y_train)

# 3. Prédictions
y_pred = lda.predict(X_test_scaled)

# 4. Évaluation
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 5. LDA peut aussi servir de RÉDUCTION DE DIMENSION (max C-1 = 9 composantes pour MNIST)
lda_reduction = LinearDiscriminantAnalysis(n_components=2)
X_reduced_2d = lda_reduction.fit_transform(X_train_scaled, y_train)
''', language="python")

    st.header("Points clés du code")
    st.markdown("""
    - `solver="svd"` : recommandé quand le nombre de features (784) est grand — ne nécessite pas d'inverser Σ directement
    - LDA a **deux usages** : classification (`.predict()`) et réduction de dimension (`.transform()`)
    - Pour la réduction, `n_components` ≤ (nombre de classes - 1), donc **max 9** pour MNIST
    - Contrairement à QDA, LDA suppose une covariance **partagée** entre classes → moins de paramètres à estimer, plus stable en haute dimension
    """)

# ----------------------------------------------------------------------
# ONGLET INTERACTIF
# ----------------------------------------------------------------------
with tab_interactif:
    st.header("Testez la LDA sur MNIST")

    col_params, col_main = st.columns([1, 3])
    with col_params:
        n_samples = st.slider("Nombre d'images (échantillon)", 500, 10000, 3000, step=500, key="lda_n")
        solver = st.selectbox("Solveur", ["svd", "lsqr", "eigen"])

    X_train, X_test, y_train, y_test, scaler = get_mnist_sample(n_samples=n_samples)

    with st.spinner("Entraînement de la LDA..."):
        lda = LinearDiscriminantAnalysis(solver=solver)
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    with col_main:
        st.metric("Accuracy sur le test set", f"{acc*100:.2f}%")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Matrice de confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Prédiction", y="Vraie classe", color="Nombre"),
            x=[str(i) for i in range(10)], y=[str(i) for i in range(10)],
        )
        st.plotly_chart(fig_cm, width='stretch')

    with col_b:
        st.subheader("Projection discriminante (2D)")
        st.caption("Projection des données sur les 2 premières directions LDA (celles qui séparent le mieux les classes).")
        lda_2d = LinearDiscriminantAnalysis(n_components=2, solver="svd")
        X_2d = lda_2d.fit_transform(X_train, y_train)
        fig_2d = px.scatter(
            x=X_2d[:, 0], y=X_2d[:, 1], color=y_train.astype(str),
            labels={"x": "LD1", "y": "LD2", "color": "Chiffre"},
            opacity=0.7,
        )
        st.plotly_chart(fig_2d, width='stretch')

    st.divider()
    st.subheader("Rapport de classification détaillé")
    report = classification_report(y_test, y_pred, output_dict=True)
    import pandas as pd
    df_report = pd.DataFrame(report).transpose().round(3)
    st.dataframe(df_report, width='stretch')
