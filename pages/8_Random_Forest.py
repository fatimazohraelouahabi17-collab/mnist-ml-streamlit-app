import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.data_loader import get_mnist_sample

st.set_page_config(page_title="Random Forest - MNIST", page_icon="🌲", layout="wide")
st.title("🌲 Random Forest — Forêt Aléatoire")

tab_theorie, tab_code, tab_interactif = st.tabs(["📐 Théorie", "💻 Code", "🎛️ Interactif"])

with tab_theorie:
    st.header("Intuition")
    st.markdown("""
    Un seul arbre de décision est instable et surapprend facilement. La
    **Random Forest** entraîne **plusieurs centaines d'arbres**, chacun sur des
    données et des features légèrement différentes (aléatoire), puis
    **moyenne leurs votes**. C'est le principe du **bagging** (bootstrap aggregating).
    """)

    st.header("Formulation mathématique")
    st.markdown("**1. Bootstrap : échantillonnage avec remise**")
    st.latex(r"""
    D_b = \{(x_i, y_i)\}_{i=1}^{n} \text{ tiré avec remise de } D, \quad b = 1, \dots, B
    """)
    st.caption("Chaque arbre b est entraîné sur un échantillon bootstrap différent (≈63% de données uniques).")

    st.markdown("**2. Sélection aléatoire de features à chaque split**")
    st.latex(r"""
    m = \sqrt{d} \quad \text{(règle par défaut pour la classification, } d = 784 \Rightarrow m \approx 28)
    """)
    st.caption("À chaque nœud, on ne considère qu'un sous-ensemble aléatoire de m features (pas toutes les 784) → décorrèle les arbres entre eux.")

    st.markdown("**3. Agrégation (vote majoritaire)**")
    st.latex(r"""
    \hat{y} = \arg\max_{c} \sum_{b=1}^{B} \mathbb{1}[\hat{y}_b(x) = c]
    """)

    st.markdown("**4. Pourquoi ça marche : réduction de variance**")
    st.latex(r"""
    \text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2
    """)
    st.caption("""
    σ² = variance d'un arbre individuel, ρ = corrélation entre arbres, B = nombre d'arbres.
    Plus les arbres sont décorrélés (ρ faible, grâce au bootstrap + sélection aléatoire de features),
    plus la variance de la moyenne diminue → modèle plus stable qu'un arbre seul.
    """)

with tab_code:
    st.header("Implémentation")
    st.code('''
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,      # nombre d'arbres (B)
    max_depth=None,        # souvent laissé libre (le bagging compense l'overfitting)
    max_features="sqrt",   # m = sqrt(784) ≈ 28 features considérées par split
    min_samples_leaf=1,
    n_jobs=-1,              # parallélise sur tous les coeurs CPU
    random_state=42,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Importance moyenne des features sur TOUS les arbres
importances = rf.feature_importances_   # shape (784,)

# Probabilités de prédiction (utile pour l'incertitude)
proba = rf.predict_proba(X_test)   # shape (n_samples, 10)
''', language="python")
    st.markdown("""
    **Points clés:**
    - `n_estimators` : plus il y en a, plus stable, mais coût de calcul linéaire en B
    - `n_jobs=-1` : les arbres sont indépendants, donc parallélisables → gain de vitesse important
    - Random Forest ne surapprend quasiment jamais en augmentant `n_estimators` (contrairement à `max_depth` sur un seul arbre)
    """)

with tab_interactif:
    st.header("Testez Random Forest sur MNIST")

    col_params, col_main = st.columns([1, 3])
    with col_params:
        n_samples = st.slider("Nombre d'images", 500, 8000, 3000, step=500, key="rf_n")
        n_estimators = st.slider("Nombre d'arbres", 10, 300, 100, step=10)
        max_depth = st.select_slider("Profondeur max", options=[5, 10, 15, 20, None], value=None)
        max_features = st.selectbox("Features par split", ["sqrt", "log2", None])

    X_train, X_test, y_train, y_test, scaler = get_mnist_sample(n_samples=n_samples)

    with st.spinner("Entraînement de la forêt..."):
        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features=max_features, n_jobs=-1, random_state=42,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    with col_main:
        st.metric("Accuracy sur le test set", f"{acc*100:.2f}%")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Matrice de confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Greens",
            labels=dict(x="Prédiction", y="Vraie classe", color="Nombre"),
            x=[str(i) for i in range(10)], y=[str(i) for i in range(10)],
        )
        st.plotly_chart(fig_cm, width='stretch')

    with col_b:
        st.subheader("Importance des pixels (heatmap 28×28)")
        importance_img = rf.feature_importances_.reshape(28, 28)
        fig_imp = px.imshow(importance_img, color_continuous_scale="Viridis",
                             title="Pixels les plus discriminants (moyenne sur tous les arbres)")
        st.plotly_chart(fig_imp, width='stretch')

    st.divider()
    st.subheader("Accuracy en fonction du nombre d'arbres")
    st.caption("Illustre pourquoi Random Forest ne surapprend pas en augmentant B")
    n_estimators_range = [5, 10, 25, 50, 100, 150, 200]
    accs = []
    for n_est in n_estimators_range:
        rf_tmp = RandomForestClassifier(n_estimators=n_est, n_jobs=-1, random_state=42)
        rf_tmp.fit(X_train, y_train)
        accs.append(accuracy_score(y_test, rf_tmp.predict(X_test)))

    fig_nest = px.line(x=n_estimators_range, y=accs, markers=True,
                        labels={"x": "Nombre d'arbres (n_estimators)", "y": "Accuracy test"},
                        title="Convergence de l'accuracy avec le nombre d'arbres")
    st.plotly_chart(fig_nest, width='stretch')

    st.divider()
    st.subheader("Random Forest vs un seul arbre")
    from sklearn.tree import DecisionTreeClassifier
    single_tree = DecisionTreeClassifier(random_state=42)
    single_tree.fit(X_train, y_train)
    acc_tree = accuracy_score(y_test, single_tree.predict(X_test))

    col_c1, col_c2 = st.columns(2)
    col_c1.metric("Un seul arbre de décision", f"{acc_tree*100:.2f}%")
    col_c2.metric("Random Forest", f"{acc*100:.2f}%", delta=f"{(acc-acc_tree)*100:+.2f} pts")
