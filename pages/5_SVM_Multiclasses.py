import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.data_loader import get_mnist_sample

st.set_page_config(page_title="SVM Multi-classes - MNIST", page_icon="🧭", layout="wide")
st.title("🧭 SVM — Classification Multi-classes")

tab_theorie, tab_code, tab_interactif = st.tabs(["📐 Théorie", "💻 Code", "🎛️ Interactif"])

with tab_theorie:
    st.header("Intuition")
    st.markdown("""
    SVM cherche l'**hyperplan séparateur** qui maximise la marge (distance)
    entre les points les plus proches de chaque classe (les **vecteurs de support**).
    Pour des données non-linéairement séparables (comme MNIST), on utilise
    l'**astuce du noyau (kernel trick)** pour projeter implicitement dans un
    espace de plus grande dimension où la séparation devient linéaire.
    """)

    st.header("Formulation mathématique (cas binaire, base)")
    st.markdown("**1. Hyperplan séparateur**")
    st.latex(r"w^T x + b = 0")

    st.markdown("**2. Marge à maximiser (SVM à marge souple)**")
    st.latex(r"""
    \min_{w,b,\xi} \; \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i
    """)
    st.latex(r"""
    \text{sous contrainte : } y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
    """)
    st.caption("ξᵢ = variables d'écart (slack) qui tolèrent des erreurs. C contrôle le compromis marge/erreurs.")

    st.markdown("**3. Kernel trick**")
    st.latex(r"""
    K(x_i, x_j) = \phi(x_i)^T \phi(x_j)
    """)
    st.markdown("Noyau RBF (le plus utilisé) :")
    st.latex(r"""
    K_{RBF}(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)
    """)
    st.caption("γ contrôle l'influence de chaque point : γ grand = frontière très locale (risque d'overfitting).")

    st.markdown("**4. Multi-classes**")
    st.markdown("""
    SVM est nativement **binaire**. Pour 10 classes (MNIST), scikit-learn utilise
    par défaut la stratégie **one-vs-one (OvO)** : entraîne C(10,2) = **45 classifieurs
    binaires**, puis vote majoritaire.
    """)

with tab_code:
    st.header("Implémentation")
    st.code('''
from sklearn.svm import SVC

# Kernel RBF (par défaut, non-linéaire)
svm = SVC(
    kernel="rbf",
    C=1.0,        # régularisation : C petit = marge large, C grand = colle aux données
    gamma="scale",  # influence de chaque point d'entraînement
    decision_function_shape="ovo",  # one-vs-one pour le multi-classes
)
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)

# Nombre de vecteurs de support par classe
print(svm.n_support_)   # ex: [45, 38, 52, ...]
''', language="python")
    st.markdown("""
    **Points clés:**
    - `kernel="linear"` pour une frontière linéaire, `"rbf"` ou `"poly"` pour non-linéaire
    - SVM peut être **lent** sur de gros datasets (complexité ~O(n²) à O(n³)) → sous-échantillonnage recommandé
    - `svm.support_vectors_` donne accès aux points qui définissent la frontière
    """)

with tab_interactif:
    st.header("Testez SVM multi-classes sur MNIST")

    col_params, col_main = st.columns([1, 3])
    with col_params:
        n_samples = st.slider("Nombre d'images", 500, 5000, 2000, step=250, key="svm_n")
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        C = st.select_slider("C (régularisation)", options=[0.01, 0.1, 1, 10, 100], value=1)
        gamma = st.select_slider("Gamma", options=["scale", "auto", 0.001, 0.01, 0.1], value="scale")

    X_train, X_test, y_train, y_test, scaler = get_mnist_sample(n_samples=n_samples)

    with st.spinner("Entraînement SVM (peut prendre quelques secondes)..."):
        svm = SVC(kernel=kernel, C=C, gamma=gamma)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    with col_main:
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Accuracy sur le test set", f"{acc*100:.2f}%")
        col_m2.metric("Nombre total de vecteurs de support", int(svm.n_support_.sum()))

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Matrice de confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Oranges",
            labels=dict(x="Prédiction", y="Vraie classe", color="Nombre"),
            x=[str(i) for i in range(10)], y=[str(i) for i in range(10)],
        )
        st.plotly_chart(fig_cm, width='stretch')

    with col_b:
        st.subheader("Vecteurs de support par classe")
        fig_sv = px.bar(x=[str(i) for i in range(10)], y=svm.n_support_,
                         labels={"x": "Chiffre", "y": "Nb vecteurs de support"})
        st.plotly_chart(fig_sv, width='stretch')

    st.divider()
    st.subheader("Frontière de décision (projetée en 2D via PCA)")
    st.caption("""
    On réduit à 2D avec PCA puis on réentraîne un SVM sur ces 2 dimensions
    uniquement, pour pouvoir visualiser la frontière (impossible directement en 784D).
    """)
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_train)
    svm_2d = SVC(kernel=kernel, C=C, gamma=gamma)
    svm_2d.fit(X_2d, y_train)

    h = 0.5
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig_boundary = go.Figure()
    fig_boundary.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h), y=np.arange(y_min, y_max, h), z=Z,
        showscale=False, opacity=0.3, colorscale="Rainbow",
    ))
    fig_boundary.add_trace(go.Scatter(
        x=X_2d[:, 0], y=X_2d[:, 1], mode="markers",
        marker=dict(color=y_train, colorscale="Rainbow", size=5, line=dict(width=0.5, color="black")),
        text=[str(c) for c in y_train],
    ))
    fig_boundary.update_layout(title="Frontières de décision SVM (espace PCA 2D)", height=550)
    st.plotly_chart(fig_boundary, width='stretch')
