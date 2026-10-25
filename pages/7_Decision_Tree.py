import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from utils.data_loader import get_mnist_sample, get_digit_image

st.set_page_config(page_title="Arbre de Décision - MNIST", page_icon="🌳", layout="wide")
st.title("🌳 Decision Tree — Arbre de Décision")

tab_theorie, tab_code, tab_interactif = st.tabs(["📐 Théorie", "💻 Code", "🎛️ Interactif"])

with tab_theorie:
    st.header("Intuition")
    st.markdown("""
    Un arbre de décision découpe l'espace des features par une **succession de
    questions binaires** ("le pixel n°245 a-t-il une intensité > 0.5 ?"), formant
    un arbre. Chaque feuille correspond à une prédiction finale (un chiffre).
    """)

    st.header("Formulation mathématique")
    st.markdown("**1. Critère de division : impureté de Gini**")
    st.latex(r"""
    Gini(t) = 1 - \sum_{c=1}^{C} p_c(t)^2
    """)
    st.caption("p_c(t) = proportion d'exemples de la classe c dans le nœud t. Gini=0 → nœud pur (une seule classe).")

    st.markdown("**Alternative : l'entropie**")
    st.latex(r"""
    Entropy(t) = -\sum_{c=1}^{C} p_c(t) \log_2 p_c(t)
    """)

    st.markdown("**2. Gain d'information d'une division**")
    st.latex(r"""
    \Delta = I(t) - \left(\frac{n_{gauche}}{n_t} I(t_{gauche}) + \frac{n_{droite}}{n_t} I(t_{droite})\right)
    """)
    st.caption("I = Gini ou Entropy. À chaque nœud, l'algorithme choisit la feature et le seuil qui MAXIMISENT ce gain.")

    st.markdown("**3. Construction récursive (algorithme CART)**")
    st.markdown("""
    1. Partir de tous les échantillons dans le nœud racine
    2. Trouver la meilleure division (feature + seuil) qui maximise Δ
    3. Créer 2 nœuds enfants, répéter récursivement
    4. S'arrêter selon un critère (profondeur max, nombre min d'échantillons par feuille, pureté atteinte)
    """)

    st.header("Avantages et limites")
    st.markdown("""
    - ✅ Interprétable (on peut visualiser l'arbre entier)
    - ✅ Ne nécessite pas de standardisation des données
    - ❌ Très sensible au surapprentissage si pas de limite de profondeur
    - ❌ Instable (petit changement dans les données → arbre très différent)
    """)

with tab_code:
    st.header("Implémentation")
    st.code('''
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    criterion="gini",       # ou "entropy"
    max_depth=10,           # limite la profondeur (évite l'overfitting)
    min_samples_leaf=5,     # nombre min d'échantillons par feuille
    random_state=42,
)
tree.fit(X_train, y_train)   # pas besoin de standardiser !
y_pred = tree.predict(X_test)

# Importance de chaque feature (pixel)
importances = tree.feature_importances_   # shape (784,)

# Visualiser l'arbre
from sklearn.tree import plot_tree
plot_tree(tree, max_depth=3, filled=True)
''', language="python")
    st.markdown("""
    **Points clés:**
    - `max_depth` est LE paramètre le plus important pour contrôler l'overfitting
    - `feature_importances_` peut être reformé en image 28×28 pour voir quels pixels sont les plus discriminants
    - Pas besoin de `StandardScaler` : les arbres ne sont pas sensibles à l'échelle
    """)

with tab_interactif:
    st.header("Testez l'arbre de décision sur MNIST")

    col_params, col_main = st.columns([1, 3])
    with col_params:
        n_samples = st.slider("Nombre d'images", 500, 8000, 3000, step=500, key="tree_n")
        max_depth = st.slider("Profondeur max", 2, 30, 10)
        criterion = st.selectbox("Critère", ["gini", "entropy"])
        min_samples_leaf = st.slider("Min échantillons par feuille", 1, 50, 5)

    X_train, X_test, y_train, y_test, scaler = get_mnist_sample(n_samples=n_samples)

    with st.spinner("Entraînement de l'arbre..."):
        tree = DecisionTreeClassifier(
            criterion=criterion, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=42,
        )
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)
        acc_train = accuracy_score(y_train, tree.predict(X_train))

    with col_main:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Accuracy Train", f"{acc_train*100:.2f}%")
        col_m2.metric("Accuracy Test", f"{acc_test*100:.2f}%")
        col_m3.metric("Écart (overfitting)", f"{(acc_train-acc_test)*100:.2f} pts",
                       delta_color="inverse")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Matrice de confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="YlOrBr",
            labels=dict(x="Prédiction", y="Vraie classe", color="Nombre"),
            x=[str(i) for i in range(10)], y=[str(i) for i in range(10)],
        )
        st.plotly_chart(fig_cm, width='stretch')

    with col_b:
        st.subheader("Importance des pixels (heatmap 28×28)")
        importance_img = tree.feature_importances_.reshape(28, 28)
        fig_imp = px.imshow(importance_img, color_continuous_scale="Reds",
                             title="Pixels les plus utilisés pour les décisions")
        st.plotly_chart(fig_imp, width='stretch')

    st.divider()
    st.subheader("Visualisation de l'arbre (3 premiers niveaux)")
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(tree, max_depth=3, filled=True, fontsize=8,
              class_names=[str(i) for i in range(10)], ax=ax)
    st.pyplot(fig)

    st.divider()
    st.subheader("Accuracy en fonction de la profondeur (analyse overfitting)")
    depths = list(range(1, 26, 2))
    train_accs, test_accs = [], []
    for d in depths:
        t = DecisionTreeClassifier(max_depth=d, random_state=42)
        t.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, t.predict(X_train)))
        test_accs.append(accuracy_score(y_test, t.predict(X_test)))

    import pandas as pd
    df_plot = pd.DataFrame({"Profondeur": depths * 2,
                             "Accuracy": train_accs + test_accs,
                             "Ensemble": ["Train"] * len(depths) + ["Test"] * len(depths)})
    fig_depth = px.line(df_plot, x="Profondeur", y="Accuracy", color="Ensemble", markers=True,
                         title="Train vs Test accuracy selon la profondeur de l'arbre")
    st.plotly_chart(fig_depth, width='stretch')
    st.caption("Si Train >> Test à grande profondeur → overfitting clairement visible.")
