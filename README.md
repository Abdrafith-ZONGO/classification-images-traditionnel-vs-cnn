# 🧠 Image Classification: Traditional ML vs Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)

## 📌 Présentation du Projet
Ce dépôt contient le code source de l'étude comparative réalisée dans le cadre du module **Réseaux de Neurones Artificiels** (Master 1 - Vision par Ordinateur et Intelligence Artificielle). 

L'objectif de ce projet est d'implémenter et de comparer deux approches de classification d'images :
1. **Pipeline 1 (Machine Learning Traditionnel + Transfer Learning)** : Extraction de caractéristiques via des modèles pré-entraînés (VGG16, AlexNet, InceptionV3), suivie d'une classification avec des algorithmes classiques optimisés (SVM, k-NN, Arbre de Décision, Naïve Bayes).
2. **Pipeline 2 (Deep Learning)** : Conception, entraînement (from scratch) et évaluation d'un réseau de neurones convolutif (CNN) personnalisé et allégé.

> 📄 **Note académique** : Le rapport complet d'analyse contenant les résultats détaillés, graphiques et conclusions se trouve ici : [`Projet_Final_Classification/Rapport_Projet_Final.md`](Projet_Final_Classification/Rapport_Projet_Final.md). (Un document réservé au professeur).

## 📊 Jeux de Données (Datasets) évalués
L'étude est menée sur 4 datasets très hétérogènes pour tester la robustesse des modèles :
* **COVID-19 X-Ray** : Radiographies pulmonaires.
* **DTD (Describable Textures Dataset)** : Catégorisation de motifs.
* **Iris Flowers** : Images de fleurs d'Iris.
* **Wildfire Satellite Images** : Détection spatiale de feux de forêt.

*(Note : Pour respecter les limites de stockage GitHub, les images sources ne sont pas incluses dans ce dépôt).*

## ⚙️ Structure du Code
L'intégralité du code exécutable se trouve dans le sous-dossier `Projet_Final_Classification/`.
```text
Projet_Final_Classification/
├── src/
│   ├── config.py             # Configuration globale (hyperparamètres)
│   ├── dataset.py            # Gestion, splits et loaders des données
│   ├── extractors.py         # Scripts d'extraction (Transfer Learning)
│   ├── models_traditional.py # GridSearch et classifieurs classiques
│   ├── models_cnn.py         # Architecture du CNN personnalisé
│   └── utils.py              # Génération des graphiques (ROC, Matrices)
├── main.py                   # Script principal (CLI)
├── Projet_Final_Notebook.ipynb # Notebook interactif d'analyse pas à pas
└── requirements.txt          # Dépendances Python
```

## 🚀 Installation & Exécution

### 1. Préparation de l'environnement
Il est recommandé d'utiliser un environnement virtuel Python 3.10+.
```bash
cd Projet_Final_Classification
pip install -r requirements.txt
```

### 2. Exécution via Terminal (CLI)
Le script `main.py` orchestre l'ensemble de l'apprentissage et des évaluations.

*   Lancer l'analyse sur un dataset ciblé (ex: Iris) avec 15 époques pour le CNN :
    ```bash
    python main.py --dataset Iris --epochs 15
    ```
*   Lancer l'analyse complète sur TOUS les jeux de données séquentiellement :
    ```bash
    python main.py --dataset all --epochs 15
    ```
*(Les modèles seront générés dans `/models` et les graphiques de performances dans `/reports`)*.

### 3. Exécution Interactive (Jupyter)
Pour visualiser les courbes d'apprentissage et matrices en temps réel :
```bash
cd Projet_Final_Classification
jupyter notebook
```
Ouvrez le fichier `Projet_Final_Notebook.ipynb` et exécutez les cellules.
