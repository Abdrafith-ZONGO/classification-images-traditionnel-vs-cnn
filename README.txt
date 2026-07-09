Classification d'Images - Apprentissage Automatique Traditionnel vs CNN
Master 1 - Vision par Ordinateur & Intelligence Artificielle

Ce projet compare deux approches de classification d'images sur 4 jeux de donnees :
1. Pipeline 1 (Traditionnel) : Extraction de features avec des modeles pre-entraines (AlexNet, VGG16, InceptionV3) suivie de classifieurs classiques (SVM, k-NN, Arbre de Decision, Naive Bayes).
2. Pipeline 2 (Apprentissage Profond) : Entrainement d'un CNN personnalise de bout en bout sur les images brutes.

---
STRUCTURE DU DOSSIER
- Projet_Final_Classification/  : Dossier contenant le code et les resultats
  - main.py                     : Script principal pour tout entrainer et evaluer
  - src/                        : Fichiers de code source (config, dataset, models...)
  - features/                   : Cache des caracteristiques extraites (fichiers .npy)
  - models/                     : Modeles entraines sauvegardes (.pkl et .pth)
  - reports/                    : Visualisations (matrices de confusion, courbes ROC, courbes d'apprentissage, graphiques comparatifs globaux)
  - Projet_Final_Notebook.ipynb : Notebook de demonstration
  - Rapport_Projet_Final.pdf    : Rapport complet du projet au format PDF
- requirements.txt              : Liste des dependances Python necessaires
- README.txt                    : Ce fichier d'explication

---
PRE-REQUIS ET CONFIGURATION
1. Installez Python 3.9+ sur votre machine.
2. Installez les dependances requises en lancant la commande :
   pip install -r requirements.txt

---
COMMENT LANCER LE PROJET
1. Ouvrez un terminal dans le dossier Projet_Final_Classification.
2. Pour lancer l'entrainement et l'evaluation complete sur tous les jeux de donnees :
   python main.py --dataset all --epochs 15

3. Pour lancer sur un jeu de donnees specifique (par exemple Iris) :
   python main.py --dataset Iris --epochs 15

4. Options du script main.py :
   --dataset : Iris, Covid19-XRAYS, Wildfire, DTD, ou all (par defaut: Iris)
   --epochs  : Nombre d'epoques d'entrainement pour le CNN (par defaut: 15)
   --force   : Forcer la re-extraction des caracteristiques (meme si les fichiers .npy existent deja)
