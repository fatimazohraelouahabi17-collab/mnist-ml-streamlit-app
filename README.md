# MNIST ML Suite

Application Streamlit que j'ai réalisée pour explorer plusieurs algorithmes de Machine Learning sur le dataset MNIST (chiffres manuscrits de 0 à 9, images de 28x28 pixels, donc 784 variables par image).

Chaque algorithme a sa propre page dans l'application, où l'on peut lire un rappel de la théorie, modifier les paramètres et voir les résultats directement.

## Algorithmes

Réduction de dimension et visualisation : PCA, t-SNE et UMAP.

Classification : LDA, QDA, KNN, SVM (multiclasses et One-Class), arbre de décision et Random Forest.

## Technologies utilisées

Python 3.11, Streamlit, scikit-learn, UMAP-learn, NumPy et Matplotlib.

## Lancer le projet

Il faut d'abord cloner le dépôt et installer les dépendances :

    git clone https://github.com/fatimazohraelouahabi17-collab/mnist-ml-streamlit-app.git
    cd mnist-ml-streamlit-app
    pip install -r requirements.txt

Ensuite on lance l'application avec :

    streamlit run app.py

Elle s'ouvre alors dans le navigateur à l'adresse http://localhost:8501.

## Organisation des fichiers

    app.py               page d'accueil
    requirements.txt     liste des packages
    pages/               une page par algorithme
    utils/data_loader.py chargement des données MNIST

## Auteure

El Ouahabi Fatima Zohra, étudiante en Master Intelligence Artificielle pour l'Économie Numérique et la Gestion (AIENG), FSJES Tanger, Université Abdelmalek Essaâdi.