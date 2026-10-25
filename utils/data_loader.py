"""
Module utilitaire pour charger et préparer les données MNIST.
Utilisé par toutes les pages de l'application.
"""

import streamlit as st
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@st.cache_data(show_spinner="Téléchargement de MNIST depuis OpenML (peut prendre 1-2 min la 1ère fois)...")
def load_mnist_raw():
    """
    Charge le dataset MNIST complet (70000 images de 28x28 = 784 pixels).
    Mis en cache par Streamlit : ne sera téléchargé qu'une seule fois.
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)
    return X, y


@st.cache_data(show_spinner="Préparation de l'échantillon de données...")
def get_mnist_sample(n_samples=5000, test_size=0.2, scale=True, random_state=42):
    """
    Retourne un sous-échantillon stratifié de MNIST, prêt à l'emploi.

    Pourquoi un sous-échantillon ?
    MNIST complet = 70000 images. Des modèles comme SVM ou QDA deviennent
    très lents (voire ingérables) sur des dizaines de milliers de points
    en 784 dimensions. On travaille donc sur un échantillon représentatif.

    Paramètres
    ----------
    n_samples : nombre total d'images à garder (avant split train/test)
    test_size : proportion réservée au test
    scale : si True, standardise les pixels (moyenne 0, écart-type 1)
            -> important pour PCA, LDA, QDA, SVM, KNN (sensibles à l'échelle)

    Retourne
    --------
    X_train, X_test, y_train, y_test, scaler (ou None si scale=False)
    """
    X, y = load_mnist_raw()

    # Échantillonnage stratifié : on garde la même proportion de chaque chiffre (0-9)
    if n_samples < len(X):
        X_sub, _, y_sub, _ = train_test_split(
            X, y,
            train_size=n_samples,
            stratify=y,
            random_state=random_state,
        )
    else:
        X_sub, y_sub = X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub,
        test_size=test_size,
        stratify=y_sub,
        random_state=random_state,
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def get_digit_image(x_row):
    """Reconvertit une ligne de pixels (784,) en image 28x28 pour affichage."""
    return x_row.reshape(28, 28)
