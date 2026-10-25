import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

from utils.data_loader import get_mnist_sample, get_digit_image

st.set_page_config(page_title="SVM One-Class - MNIST", page_icon="🎯", layout="wide")
st.title("🎯 One-Class SVM — Détection d'anomalies")

tab_theorie, tab_code, tab_interactif = st.tabs(["📐 Théorie", "💻 Code", "🎛️ Interactif"])

with tab_theorie:
    st.header("Intuition")
    st.markdown("""
    Contrairement au SVM classique (qui sépare plusieurs classes), le
    **One-Class SVM** apprend à partir d'**une seule classe** ("normale") et
    cherche une frontière qui **entoure** cette classe dans l'espace des features.
    Tout point en dehors de cette frontière est considéré comme une **anomalie**
    (ou "nouveauté").

    **Application sur MNIST** : on entraîne le modèle uniquement sur les images
    d'**un seul chiffre** (ex : que des "0"). Le modèle doit ensuite détecter
    que les autres chiffres (1, 2, 3...) sont des "anomalies" par rapport à ce qu'il a appris.
    """)

    st.header("Formulation mathématique")
    st.markdown("**Formulation ν-SVM (Schölkopf et al.)**")
    st.latex(r"""
    \min_{w, \rho, \xi} \; \frac{1}{2}\|w\|^2 - \rho + \frac{1}{\nu n}\sum_{i=1}^{n} \xi_i
    """)
    st.latex(r"""
    \text{sous contrainte : } w^T \phi(x_i) \geq \rho - \xi_i, \quad \xi_i \geq 0
    """)
    st.caption("""
    On cherche l'hyperplan qui sépare les données de l'origine (dans l'espace
    transformé φ) avec une marge maximale ρ. Le paramètre **ν ∈ (0,1]** contrôle :
    - une borne supérieure sur la fraction d'outliers tolérés dans l'entraînement
    - une borne inférieure sur la fraction de vecteurs de support
    """)

    st.markdown("**Fonction de décision**")
    st.latex(r"""
    f(x) = \text{sign}\left(\sum_{i} \alpha_i K(x_i, x) - \rho\right)
    """)
    st.caption("f(x) > 0 → point normal (inlier) ; f(x) < 0 → anomalie (outlier)")

with tab_code:
    st.header("Implémentation")
    st.code('''
from sklearn.svm import OneClassSVM

# Entraîner UNIQUEMENT sur les images d'une classe "normale" (ex: le chiffre 0)
X_normal = X_train[y_train == 0]

ocsvm = OneClassSVM(
    kernel="rbf",
    nu=0.05,        # proportion attendue d'outliers/erreurs tolérées (0 < nu <= 1)
    gamma="scale",
)
ocsvm.fit(X_normal)

# Prédire sur un mélange de "0" et d'autres chiffres
predictions = ocsvm.predict(X_test)   # +1 = normal, -1 = anomalie
scores = ocsvm.decision_function(X_test)  # score continu (plus c'est bas, plus c'est anormal)
''', language="python")
    st.markdown("""
    **Points clés:**
    - `nu` est le paramètre le plus important : petit `nu` → frontière plus large (tolérante), grand `nu` → frontière plus stricte
    - Retourne **+1 / -1**, pas les classes d'origine
    - Utile pour : détection de fraude, contrôle qualité industriel, détection d'écriture atypique...
    """)

with tab_interactif:
    st.header("Testez le One-Class SVM sur MNIST")

    col_params, col_main = st.columns([1, 3])
    with col_params:
        n_samples = st.slider("Nombre d'images (pool total)", 1000, 8000, 3000, step=500, key="ocsvm_n")
        digit_normal = st.selectbox("Chiffre considéré comme 'normal'", list(range(10)), index=0)
        nu = st.slider("nu (tolérance aux outliers)", 0.01, 0.5, 0.05, step=0.01)
        gamma = st.select_slider("Gamma", options=["scale", "auto", 0.001, 0.01, 0.1], value="scale")

    X_train, X_test, y_train, y_test, scaler = get_mnist_sample(n_samples=n_samples)

    # Entraînement uniquement sur le chiffre "normal"
    X_normal_train = X_train[y_train == digit_normal]

    with st.spinner("Entraînement du One-Class SVM..."):
        ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        ocsvm.fit(X_normal_train)

        # Test sur TOUT le test set (mélange normal + anomalies)
        y_pred_raw = ocsvm.predict(X_test)  # +1 ou -1
        y_true_binary = np.where(y_test == digit_normal, 1, -1)  # vérité terrain

    with col_main:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true_binary, y_pred_raw)
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Images 'normales' à l'entraînement", len(X_normal_train))
        col_m2.metric("Accuracy (normal vs anomalie)", f"{acc*100:.2f}%")
        col_m3.metric("Vecteurs de support", len(ocsvm.support_))

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Rapport de classification (normal vs anomalie)")
        report = classification_report(
            y_true_binary, y_pred_raw,
            target_names=["Anomalie (-1)", f"Normal = chiffre {digit_normal} (+1)"],
            output_dict=True,
        )
        import pandas as pd
        st.dataframe(pd.DataFrame(report).transpose().round(3), width='stretch')

    with col_b:
        st.subheader("Distribution des scores de décision")
        scores = ocsvm.decision_function(X_test)
        df_scores = np.column_stack([scores, np.where(y_test == digit_normal, "Normal", "Anomalie")])
        fig_hist = px.histogram(
            x=scores.astype(float), color=df_scores[:, 1],
            nbins=50, barmode="overlay", opacity=0.6,
            labels={"x": "Score de décision", "color": "Type"},
            title="Scores : positif = normal, négatif = anomalie",
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_hist, width='stretch')

    st.divider()
    st.subheader("Exemples d'images classées 'anomalies' malgré être un vrai chiffre différent")
    anomalies_idx = np.where((y_pred_raw == -1) & (y_test != digit_normal))[0]
    false_anomalies_idx = np.where((y_pred_raw == -1) & (y_test == digit_normal))[0]

    st.markdown(f"**Anomalies correctement détectées** (vrai chiffre ≠ {digit_normal}) :")
    if len(anomalies_idx) > 0:
        cols = st.columns(min(8, len(anomalies_idx)))
        for i, idx in enumerate(anomalies_idx[:8]):
            with cols[i]:
                st.image(get_digit_image(X_test[idx]), caption=f"vrai={y_test[idx]}", width=80, clamp=True)
    else:
        st.info("Aucune anomalie trouvée avec ces paramètres.")

    st.markdown(f"**Faux positifs** (vrais '{digit_normal}' classés comme anomalies) :")
    if len(false_anomalies_idx) > 0:
        cols = st.columns(min(8, len(false_anomalies_idx)))
        for i, idx in enumerate(false_anomalies_idx[:8]):
            with cols[i]:
                st.image(get_digit_image(X_test[idx]), caption=f"'{digit_normal}' raté", width=80, clamp=True)
    else:
        st.success("Aucun faux positif — tous les vrais exemples sont bien reconnus comme normaux !")
