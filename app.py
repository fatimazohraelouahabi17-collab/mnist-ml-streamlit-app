import streamlit as st

st.set_page_config(
    page_title="MNIST ML Suite",
    page_icon="🔢",
    layout="wide",
)

st.title("🔢 MNIST Machine Learning Suite")

st.markdown("""
Bienvenue dans cette application pédagogique explorant plusieurs algorithmes
de Machine Learning appliqués au dataset **MNIST** (chiffres manuscrits 0-9,
images 28×28 pixels = 784 features).

### 📚 Algorithmes disponibles (menu à gauche)

| Page | Type | Description |
|---|---|---|
| **PCA** | Réduction de dimension | Analyse en Composantes Principales |
| **t-SNE & UMAP** | Réduction de dimension | Visualisation non-linéaire |
| **LDA** | Classification | Analyse Discriminante Linéaire |
| **QDA** | Classification | Analyse Discriminante Quadratique |
| **KNN** | Classification | K plus proches voisins |
| **SVM** | Classification | Machines à vecteurs de support |
| **Random Forest** | Classification | Forêt aléatoire d'arbres de décision |

### 🗂️ Structure de chaque page
Chaque page suit la même structure pédagogique :
1. **Théorie** — les fondements mathématiques (formules, intuition)
2. **Code** — l'implémentation Python / scikit-learn commentée
3. **Interactif** — testez l'algorithme en direct sur MNIST avec vos propres paramètres

### ⚙️ Note technique
Pour des raisons de performance, les modèles travaillent sur un **sous-échantillon
stratifié** de MNIST (nombre configurable dans chaque page), plutôt que sur
les 70 000 images complètes. Les données sont mises en cache après le premier
chargement.

👈 **Choisissez une page dans le menu de gauche pour commencer.**
""")

st.info("💡 Astuce : le premier chargement de MNIST peut prendre 1-2 minutes (téléchargement depuis OpenML). Ensuite tout est mis en cache.")
