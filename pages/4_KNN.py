import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

from utils.data_loader import get_mnist_sample, get_digit_image

st.set_page_config(page_title="KNN - MNIST", page_icon="📍", layout="wide")
st.title("📍 KNN — K Plus Proches Voisins")

tab_theorie, tab_code, tab_interactif = st.tabs(["📐 Théorie", "💻 Code", "🎛️ Interactif"])

with tab_theorie:
    st.header("Intuition")
    st.markdown("""
    KNN est un algorithme **non-paramétrique et paresseux (lazy learning)** :
    il ne construit aucun modèle à l'entraînement. Pour classifier un nouveau
    point, il regarde simplement **les K points les plus proches** dans le
    jeu d'entraînement et vote la classe majoritaire.
    """)

    st.header("Formulation mathématique")
    st.markdown("**1. Distance entre deux images (distance euclidienne)**")
    st.latex(r"""
    d(x, x_i) = \sqrt{\sum_{j=1}^{784} (x_j - x_{i,j})^2} = \|x - x_i\|_2
    """)
    st.caption("D'autres distances existent : Manhattan (L1), Minkowski, cosinus...")

    st.markdown("**2. Sélection des K voisins**")
    st.latex(r"""
    N_K(x) = \{ i \; : \; d(x, x_i) \text{ parmi les K plus petites distances} \}
    """)

    st.markdown("**3. Vote majoritaire**")
    st.latex(r"""
    \hat{y} = \arg\max_{c} \sum_{i \in N_K(x)} \mathbb{1}[y_i = c]
    """)

    st.header("Choix de K : biais-variance")
    st.markdown("""
    - **K petit** (ex: K=1) → modèle très flexible, colle au bruit → **overfitting**, forte variance
    - **K grand** → modèle plus lisse, généralise mieux mais peut ignorer des nuances → **underfitting**, fort biais
    - En pratique, on teste plusieurs K par validation croisée
    """)

with tab_code:
    st.header("Implémentation")
    st.code('''
from sklearn.neighbors import KNeighborsClassifier

# 1. Créer le modèle
knn = KNeighborsClassifier(
    n_neighbors=5,        # K
    metric="euclidean",   # ou "manhattan", "minkowski"...
    weights="uniform",    # ou "distance" (pondère par 1/distance)
)

# 2. "Entraîner" (en réalité : juste stocker les données)
knn.fit(X_train_scaled, y_train)

# 3. Prédire
y_pred = knn.predict(X_test_scaled)

# 4. Trouver les voisins d'un point précis (utile pour visualiser)
distances, indices = knn.kneighbors(X_test_scaled[[0]], n_neighbors=5)
''', language="python")
    st.markdown("""
    **Points clés:**
    - KNN est **sensible à l'échelle** → standardisation indispensable
    - `kneighbors()` retourne les indices des voisins → utile pour visualiser "pourquoi" une prédiction a été faite
    - Coût de prédiction élevé (calcule les distances à TOUT le train set à chaque prédiction) → lent sur gros datasets
    """)

with tab_interactif:
    st.header("Testez KNN sur MNIST")

    col_params, col_main = st.columns([1, 3])
    with col_params:
        n_samples = st.slider("Nombre d'images", 500, 8000, 3000, step=500, key="knn_n")
        k = st.slider("K (nombre de voisins)", 1, 30, 5)
        metric = st.selectbox("Métrique de distance", ["euclidean", "manhattan", "chebyshev"])
        weights = st.radio("Pondération", ["uniform", "distance"])

    X_train, X_test, y_train, y_test, scaler = get_mnist_sample(n_samples=n_samples)

    with st.spinner("Entraînement KNN..."):
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
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
        st.subheader("Accuracy en fonction de K")
        st.caption("Calculé sur un sous-échantillon pour la rapidité")
        k_range = list(range(1, 21, 2))
        accs = []
        X_tr_small = X_train[:1500]
        y_tr_small = y_train[:1500]
        X_te_small = X_test[:500]
        y_te_small = y_test[:500]
        for k_val in k_range:
            knn_tmp = KNeighborsClassifier(n_neighbors=k_val, metric=metric)
            knn_tmp.fit(X_tr_small, y_tr_small)
            accs.append(accuracy_score(y_te_small, knn_tmp.predict(X_te_small)))
        fig_k = px.line(x=k_range, y=accs, markers=True,
                         labels={"x": "K", "y": "Accuracy"},
                         title="Impact du choix de K")
        st.plotly_chart(fig_k, width='stretch')

    st.divider()
    st.subheader("Visualiser les voisins d'une image test")
    idx = st.slider("Choisir une image du test set", 0, len(X_test) - 1, 0)
    distances, indices = knn.kneighbors(X_test[[idx]], n_neighbors=5)

    st.markdown(f"**Image à classifier** (vraie classe : {y_test[idx]}, prédite : {y_pred[idx]})")
    cols = st.columns(6)
    with cols[0]:
        st.image(get_digit_image(X_test[idx]), caption="Requête", width=100, clamp=True)
    for i, (neighbor_idx, dist) in enumerate(zip(indices[0], distances[0])):
        with cols[i + 1]:
            st.image(get_digit_image(X_train[neighbor_idx]),
                      caption=f"Voisin {i+1}\nclasse={y_train[neighbor_idx]}\nd={dist:.1f}",
                      width=100, clamp=True)
