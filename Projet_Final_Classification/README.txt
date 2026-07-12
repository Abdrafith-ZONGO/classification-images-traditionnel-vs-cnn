Classification d'Images - Pipelines ML Traditionnel vs Deep Learning
Projet Final - Réseaux de Neurones Artificiels
Master M1 - Vision par Ordinateur et Intelligence Artificielle

-------------------------------------------------------------------------------
1. PRESENTATION DU PROJET
-------------------------------------------------------------------------------
Ce projet implémente et compare deux approches de classification d'images :
- Pipeline 1 (Traditionnel) : Extraction de caractéristiques par réseaux 
  de neurones pré-entraînés (VGG16, AlexNet, InceptionV3) suivie de 
  classifieurs classiques (SVM, k-NN, Arbre de Décision, Naïve Bayes).
- Pipeline 2 (Deep Learning) : Entraînement direct d'un CNN personnalisé
  (4 couches de convolution, batch normalization, pooling et dropout) 
  sur les images brutes.

Les évaluations sont menées sur quatre jeux de données :
- COVID-19 X-Ray (images de radiographies pulmonaires CT)
- DTD (jeu de données d'animaux)
- Iris Flowers (images de fleurs d'Iris)
- Wildfire Satellite Images (images satellites de feux de forêt)

-------------------------------------------------------------------------------
2. STRUCTURE DU PROJET
-------------------------------------------------------------------------------
Projet_Final_Classification/
├── src/
│   ├── config.py             : Fichier de configuration globale
│   ├── dataset.py            : Gestion et split des données (loaders, filtrage)
│   ├── extractors.py         : Extraction de caractéristiques
│   ├── models_traditional.py : Classifieurs classiques (GridSearchCV)
│   ├── models_cnn.py         : Définition et entraînement du CNN personnalisé
│   └── utils.py              : Traçage des courbes ROC, matrices et courbes d'apprentissage
├── features/                 : Fichiers de caractéristiques extraites (.npy)
├── models/                   : Sauvegarde des modèles (.pth et .pkl)
├── reports/                  : Rapports de performance et graphiques générés
├── main.py                   : Script principal d'orchestration (CLI)
├── Projet_Final_Notebook.ipynb : Notebook de démonstration et d'interprétation
├── requirements.txt          : Liste des dépendances logicielles
└── README.txt                : Ce fichier d'instructions (le présent fichier)

-------------------------------------------------------------------------------
3. INSTALLATION DES DEPENDANCES
-------------------------------------------------------------------------------
Il est recommandé d'utiliser un environnement virtuel Python 3.10+ propre.
Pour installer les bibliothèques requises, exécutez la commande suivante :

    pip install -r requirements.txt

-------------------------------------------------------------------------------
4. EXECUTION DU PIPELINE (LIGNE DE COMMANDE)
-------------------------------------------------------------------------------
Le script principal "main.py" permet de lancer les expériences. Par défaut, il 
s'exécute sur le dataset Iris.

Pour lancer le pipeline sur un jeu de données spécifique (ex: Iris) :
    python main.py --dataset Iris --epochs 15

Pour exécuter tous les jeux de données séquentiellement :
    python main.py --dataset all --epochs 15

Options disponibles :
    --dataset {all, Covid19-XRAYS, DTD, Iris, Wildfire} : Sélection du dataset
    --epochs N : Nombre d'époques pour l'entraînement du CNN personnalisé
    --force : Force l'extraction et remplace les caractéristiques enregistrées

Les modèles entraînés sont enregistrés dans le dossier "models/", les features
dans le dossier "features/", et les graphiques (matrices de confusion, courbes 
ROC et courbes d'apprentissage) dans le dossier "reports/".

-------------------------------------------------------------------------------
5. INTERACTION VIA LE NOTEBOOK JUPYTER
-------------------------------------------------------------------------------
Pour une démonstration interactive pas à pas avec interprétations détaillées :
1. Lancez le serveur Jupyter Notebook :
    jupyter notebook
2. Ouvrez le fichier "Projet_Final_Notebook.ipynb".
3. Exécutez les cellules séquentiellement pour voir les visualisations s'afficher.
