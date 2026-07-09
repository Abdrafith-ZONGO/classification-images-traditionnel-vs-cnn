# Rapport de Projet : Classification d'Images

**Etudiant** : Abdrafith ZONGO  
**Depot GitHub** : [classification-images-traditionnel-vs-cnn](https://github.com/Abdrafith-ZONGO/classification-images-traditionnel-vs-cnn.git)  
**Module** : Reseaux de Neurones Artificiels  

---

## 1. Introduction
La classification d'images consiste a attribuer automatiquement une ou plusieurs etiquettes a une image en fonction de son contenu visuel. Ce domaine a ete revolutionne par l'arrivee de l'apprentissage profond (Deep Learning), mais les approches de Machine Learning traditionnel restent tres pertinentes dans de nombreux contextes applicatifs. 

Ce projet propose de comparer systematiquement deux approches fondamentales :
- **Le Pipeline 1 (Traditionnel)** : on extrait les caracteristiques (features) des images a l'aide de modeles convolutifs profonds pre-entraines sur ImageNet (AlexNet, VGG16, InceptionV3), puis on entraine des classifieurs classiques (SVM, k-NN, Arbre de Decision, Naive Bayes) sur ces vecteurs.
- **Le Pipeline 2 (Apprentissage Profond)** : on entraine un reseau de neurones convolutif (CNN) personnalise directement sur les images brutes, sans extraction intermediaire de caracteristiques.

Cette comparaison est realisee sur quatre jeux de donnees varies : DTD (textures), Iris Flowers (fleurs), COVID-19 X-Ray (radiographies pulmonaires) et Wildfire Satellite Images (images satellites d'incendies).

---

## 2. Problematique
L'apprentissage profond offre des precisions exceptionnelles, mais necessite d'importantes ressources de calcul (GPU) et des bases de donnees de grande taille pour eviter le surapprentissage (overfitting). A l'inverse, l'apprentissage traditionnel sur caracteristiques pre-extraites est extremement leger et rapide a entrainer.

Ce projet cherche a repondre aux questions suivantes :
- Quel est le compromis exactitude / temps de calcul entre un CNN entraine a partir de zero et des classifieurs classiques couples a du transfert d'apprentissage ?
- Quelle approche est la plus viable selon la taille du dataset et les contraintes de calcul du materiel cible (notamment en utilisation CPU) ?

---

## 3. Jeux de Donnes Utilises
Notre protocole experimental s'appuie sur quatre bases d'images :
- **Iris Flowers** (421 images, 3 classes) : Petit jeu de donnees contenant des images de trois varietes de fleurs d'Iris (Setosa, Versicolor, Virginica).
- **COVID-19 X-Ray** (743 images, 2 classes) : Images medicales de radiographies thoraciques reparties en deux classes (CT_COVID et CT_NonCOVID).
- **Wildfire Satellite Images** (1 832 images, 2 classes) : Images satellites de zones forestieres classees selon la presence d'incendie (fire et nofire).
- **DTD** (540 images, 9 classes) : Echantillon de la base Describable Textures Dataset comprenant 9 categories d'animaux sauvages (antelope, badger, butterfly, cat, chimpanzee, cow, dragonfly, eagle, elephant).

---

## 4. Architecture Globale du Projet

Le projet est organise de facon modulaire pour separer la configuration, le chargement des donnees, l'extraction de caracteristiques, l'entrainement des modeles et l'analyse des resultats.

### 4.1. Arborescence des Fichiers
```
Projet_Final_Classification/
│
├── main.py                     # Script d'orchestration global
├── requirements.txt            # Liste des dependances Python necessaires
├── Rapport_Projet_Final.pdf    # Rapport complet au format PDF
├── Projet_Final_Notebook.ipynb # Notebook interactif pour la demonstration
│
├── src/                        # Code source du projet
│   ├── config.py               # Centralisation des parametres et constantes
│   ├── dataset.py              # Chargement, nettoyage et pretraitement des images
│   ├── extractors.py           # Chargement des reseaux de neurones et extraction de features
│   ├── models_traditional.py   # Definition et entrainement des classifieurs classiques
│   ├── models_cnn.py           # Definition de l'architecture et boucle du CNN perso
│   └── utils.py                # Fonctions d'affichage et de sauvegarde des graphiques
│
├── features/                   # Cache des caracteristiques extraites (fichiers .npy)
├── models/                     # Modeles entraines sauvegardes (.pkl et .pth)
└── reports/                    # Graphiques, courbes ROC et tableaux de resultats
```

### 4.2. Role des Fichiers et Fonctions Principales

#### 4.2.1. Fichier `src/config.py`
Ce fichier centralise tous les parametres globaux pour garantir la reproductibilite des tests.
- `SEED` : Graine aleatoire (fixee a 42) pour que les separations de donnees et les initialisations de reseaux soient les memes a chaque execution.
- `DEVICE` : Selection automatique de la carte graphique (CUDA GPU) si elle est disponible, sinon utilisation du processeur (CPU).
- `DATASET_PATHS` et `DATASET_CLASSES` : Chemins vers les bases de donnees et listes des repertoires autorises. Cela permet d'exclure programmatiquement les sous-dossiers parasites (comme le dossier "INF5082" pour Iris).
- `TAILLE_IMAGE_VGG_ALEX` (224), `TAILLE_IMAGE_INCEPTION` (299), `TAILLE_IMAGE_CNN` (128) : Tailles de redimensionnement requises pour chaque architecture.

#### 4.2.2. Fichier `src/dataset.py`
Il gere la preparation des images sous forme de loaders PyTorch.
- `obtenir_transforms(taille_image, augmentation)` : Prepare les transformations d'images (redimensionnement, normalisation ImageNet). Si l'augmentation est activee (pour l'entrainement du CNN), on ajoute des retournements horizontaux et des rotations aleatoires pour eviter le surapprentissage.
- `ImageDatasetCustom` : Classe de Dataset personnalisee chargee de lire les images sur le disque en mode RGB.
- `collecter_donnees_dataset(nom_dataset)` : Parcourt les dossiers valides, verifie l'integrite de chaque image (`PIL.Image.verify()`) et renvoie les chemins de fichiers propres.
- `preparer_loaders(nom_dataset, taille_image)` : Applique un decoupage stratifie (stratified split) de 70% pour l'entrainement et 30% pour le test, puis cree les chargeurs (`DataLoader`) correspondants.

#### 4.2.3. Fichier `src/extractors.py`
Gere l'extraction des caracteristiques grace aux modeles pre-entraines de PyTorch.
- `charger_modele_preentraine(nom_modele)` : Telecharge AlexNet, VGG16 ou InceptionV3 pre-entraines sur ImageNet, puis remplace leur derniere couche par `nn.Identity()` pour desactiver la classification.
- `extraire_features(modele, loader)` : Fait passer les images dans le reseau en mode evaluation (sans gradients) pour en recuperer les représentations vectorielles.
- `obtenir_features_dataset(nom_dataset, nom_modele)` : Coordonne le processus et stocke les vecteurs sous forme de fichiers binaire `.npy` (ex: `feat_train_Iris_VGG16.npy`). Si les fichiers existent deja sur le disque, ils sont recharges directement sans refaire les calculs.

#### 4.2.4. Fichier `src/models_traditional.py`
Contient la logique d'apprentissage classique sur les features extraites.
- `obtenir_grille_parametres(nom_classifieur)` : Contient les dictionnaires d'hyperparametres a tester.
- `instancier_classifieur_base(nom_classifieur)` : Instancie le SVM, le k-NN, l'Arbre de Decision ou le Naive Bayes de scikit-learn.
- `entrainer_evaluer_classique(...)` : Normalise les donnees avec un `StandardScaler`, puis effectue une recherche sur grille avec validation croisee (`GridSearchCV` en 3 folds) pour determiner les meilleurs reglages du classifieur. Sauvegarde le modele entraine sous format `.pkl`.

#### 4.2.5. Fichier `src/models_cnn.py`
Contient le code de notre reseau convolutif personnalise.
- `CNNPerso` : Reseau construit a partir de zero avec 4 couches convolutives.
- `entrainer_evaluer_cnn(...)` : Entraine le reseau sur les images brutes pendant un nombre d'epoques donne (par defaut 15) en utilisant l'optimiseur Adam et la fonction de perte CrossEntropyLoss. Sauvegarde les poids finaux du modele dans un fichier `.pth`.

#### 4.2.6. Fichier `src/utils.py`
Gere la creation automatique de toutes les visualisations graphiques.
- `tracer_courbes_apprentissage` : Genere les courbes de perte (Loss) et d'exactitude (Accuracy) pour l'entrainement du CNN.
- `tracer_matrice_confusion` : Dessine et sauvegarde les matrices de confusion sous forme de cartes thermiques avec Seaborn.
- `tracer_courbe_roc` : Trace les courbes ROC et calcule l'AUC (gère le cas binaire et l'approche One-vs-Rest pour le multi-classes).
- `tracer_comparaison_globale` : Genere les deux graphiques de synthese de fin de projet (graphique par dataset et classement final).

---

## 5. Methodologie et Pipelines de Classification

```
   PIPELINE 1 : Apprentissage Traditionnel
   Images Brutes -> Pretraitement -> Modeles pre-entraines -> Fichiers .npy -> StandardScaler -> GridSearchCV (Classifieurs) -> Modele Final .pkl

   PIPELINE 2 : Apprentissage Profond
   Images Brutes -> Pretraitement (avec Data Augmentation) -> CNN Personnalise (4 blocs Conv) -> Entrainement (CrossEntropy/Adam) -> Modele Final .pth
```

### 5.1. Protocoles de Pretraitement
Chaque image subit les operations suivantes avant d'entrer dans un modele :
- Redimensionnement selon le modele cible (224x224, 299x299 ou 128x128).
- Conversion en tenseur PyTorch.
- Normalisation basee sur la moyenne `[0.485, 0.456, 0.406]` et l'ecart-type `[0.229, 0.224, 0.225]` d'ImageNet.

### 5.2. Justifications Techniques
- **StandardScaler pour le SVM et le k-NN** : Les algorithmes calculant des distances (comme le k-NN ou le SVM a noyaux) sont tres sensibles a l'echelle des caracteristiques. Centrer et reduire les caracteristiques garantit que toutes les dimensions du vecteur contribuent de maniere equitable au calcul.
- **Noyau RBF pour le SVM** : Le noyau RBF (Radial Basis Function) permet de projeter nos vecteurs de caracteristiques de maniere non lineaire dans un espace de dimension infinie, facilitant ainsi la separation de classes complexes qui ne seraient pas separables avec une simple frontiere lineaire.
- **Optimisation par Validation Croisee (GridSearchCV)** : Elle permet de trouver de maniere rigoureuse le meilleur compromis de parametres (comme le nombre de voisins $k$ pour le k-NN ou la force de regularisation $C$ pour le SVM) sans risquer de surapprendre sur l'ensemble de test, puisque les performances sont validees sur des sous-ensembles d'entrainement.

### 5.3. Conception du CNN Personnalise
L'architecture de notre `CNNPerso` est definie ainsi :
- **Bloc 1** : Conv2D (3 vers 32 filtres, 3x3) + BatchNorm + ReLU + MaxPool (2x2)
- **Bloc 2** : Conv2D (32 vers 64 filtres, 3x3) + BatchNorm + ReLU + MaxPool (2x2)
- **Bloc 3** : Conv2D (64 vers 128 filtres, 3x3) + BatchNorm + ReLU + MaxPool (2x2)
- **Bloc 4** : Conv2D (128 vers 256 filtres, 3x3) + BatchNorm + ReLU + MaxPool (2x2)
- **Couche Fully Connected** : Aplatissement (Flatten) + Lineaire (512 neurones) + BatchNorm + ReLU + Dropout (50%) + Lineaire de sortie (taille egale au nombre de classes).

*Justification de l'architecture* : L'augmentation progressive de la profondeur (32 a 256 filtres) permet de capturer des formes de plus en plus abstraites. La Batch Normalization stabilise les valeurs a chaque couche pour accelerer la convergence, et la couche de Dropout a 50% evite que le classifieur dense n'apprenne les images d'entrainement par coeur (surapprentissage).

---

## 6. Resultats Experimentaux et Analyses

L'ensemble des modeles a ete entraine et evalue de maniere rigoureuse. Les resultats ci-dessous decoulent des executions completes stockees dans `reports/comparaison_globale.csv`.

### 6.1. Tableau Recapitulatif des Performances

| Dataset | Pipeline | Modele | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Temps Entr. (s) | Parametres |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Iris** | Pipeline 1 | SVM (via AlexNet) | 57.5% | 55.5% | 57.5% | 56.1% | 70.77 | 61,100,840 |
| | Pipeline 1 | k-NN (via VGG16) | **66.9%** | 78.2% | 66.9% | 56.4% | 2.28 | 138,357,544 |
| | Pipeline 2 | CNN Personnalise | 63.8% | 59.6% | 63.8% | 58.9% | 249.01 | 8,781,059 |
| **Covid19** | Pipeline 1 | SVM (via InceptionV3) | **82.5%** | 82.7% | 82.5% | 82.5% | 55.11 | 27,161,264 |
| | Pipeline 1 | Naive Bayes (via InceptionV3) | 76.2% | 76.9% | 76.2% | 75.9% | 0.46 | 27,161,264 |
| | Pipeline 2 | CNN Personnalise | 76.7% | 79.0% | 76.7% | 75.9% | 326.65 | 8,780,546 |
| **Wildfire**| Pipeline 1 | SVM (via AlexNet) | 99.3% | 99.3% | 99.3% | 99.3% | 321.12 | 61,100,840 |
| | Pipeline 1 | k-NN (via VGG16) | **99.5%** | 99.5% | 99.5% | 99.5% | 20.82 | 138,357,544 |
| | Pipeline 2 | CNN Personnalise | 95.3% | 95.4% | 95.3% | 95.3% | 676.31 | 8,780,546 |
| **DTD** | Pipeline 1 | SVM (via InceptionV3) | **98.8%** | 98.8% | 98.8% | 98.8% | 22.49 | 27,161,264 |
| | Pipeline 1 | k-NN (via InceptionV3) | **98.8%** | 98.8% | 98.8% | 98.8% | 3.20 | 27,161,264 |
| | Pipeline 2 | CNN Personnalise | 61.7% | 65.3% | 61.7% | 62.0% | 323.09 | 8,784,137 |

### 6.2. Analyse Detaillee par Dataset

#### 6.2.1. Dataset Iris Flowers
Le dataset Iris possede un faible nombre d'images (421 au total). Le pipeline traditionnel prend une avance nette, notamment le **k-NN couple a VGG16** (Accuracy = **66.9%**). Le classifieur tire avantage des descripteurs tres puissants de VGG16 deja entraine sur ImageNet. Notre CNN personnalise obtient un score de **63.8%**, ce qui est une bonne performance mais montre qu'il a du mal a generaliser par manque de donnees pour s'entrainer correctement a partir de zero.

**Interpretation graphique (Matrice de Confusion du CNN)** :
L'image ci-dessous illustre les predictions du CNN. On observe que le modele confond legerement les classes Versicolor et Virginica (qui se ressemblent enormement visuellement), mais identifie parfaitement la classe Setosa.
![Matrice de Confusion CNN Iris](reports/confusion_CNN_Personnalisé_Iris.png)

**Interpretation graphique (Courbe d'apprentissage CNN)** :
![Courbe Apprentissage Iris](reports/courbe_apprentissage_Iris.png)
La courbe montre une belle baisse de la perte d'entrainement, mais une stagnation sur le test (overfitting naissant du au petit nombre d'images).

#### 6.2.2. Dataset COVID-19 X-Ray
Sur ce dataset de 743 radiographies thoraciques, le **SVM combine aux caractéristiques d'InceptionV3** produit le meilleur resultat avec **82.5%** d'accuracy. Le CNN personnalise atteint quant a lui **76.7%**. C'est un score eleve qui prouve que notre petit CNN a reussi a developper des filtres convolutionnels adaptes a la detection d' anomalies dans les tissus pulmonaires.

**Interpretation graphique (Matrice de Confusion du SVM - InceptionV3)** :
Ce modele de Machine Learning excelle a distinguer les poumons sains des poumons infectes par le Covid-19, comme le prouve sa diagonale fortement marquee.
![Matrice de Confusion SVM Covid19](reports/confusion_SVM_via_InceptionV3_Covid19-XRAYS.png)

**Interpretation graphique (Courbe d'apprentissage CNN)** :
Ici, on voit que la courbe de Loss de test diminue conjointement avec celle de train avant de legerement remonter a la 15eme epoque. L'accuracy monte a pres de 80%.
![Courbe d'apprentissage Covid19](reports/courbe_apprentissage_Covid19-XRAYS.png)

#### 6.2.3. Dataset Wildfire Satellite Images
Les images satellites presentent des contrastes de couleur tres marques entre les zones brulees (couleurs charbon / orange) et les zones vertes de foret. Toutes nos configurations obtiennent donc d'excellents scores. Le **k-NN applique aux features de VGG16** prend la premiere place avec **99.5%** d'accuracy, tandis que notre CNN personnalise atteint **95.3%**. Le CNN montre ici qu'il est capable d'apprendre efficacement si on lui donne une base de donnees de taille convenable (1 832 images).

**Interpretation graphique (Courbe ROC du k-NN - VGG16)** :
La courbe ROC frole le coin superieur gauche, ce qui donne une aire sous la courbe (AUC) quasiment parfaite de 1.00. Cela signifie que le modele discrimine sans aucune erreur les images d'incendies des images de foret saine.
![Courbe ROC kNN Wildfire](reports/roc_k-NN_via_VGG16_Wildfire.png)

**Interpretation graphique (Matrice de Confusion du CNN)** :
Le CNN classe la tres grande majorite des images correctement, avec seulement de tres rares faux positifs (zones non brulees classees comme incendie, souvent a cause d'ombres ou de sols arides).
![Matrice de Confusion CNN Wildfire](reports/confusion_CNN_Personnalisé_Wildfire.png)

#### 6.2.4. Dataset DTD
Ce dataset represente le cas de figure le plus difficile : 9 classes d'animaux sauvages complexes pour seulement 540 images (soit 60 images par classe). Les extracteurs pre-entraines d'**InceptionV3 associes a un SVM** resolvent la tache presque parfaitement avec **98.8%** de precision. A l'inverse, le CNN personnalise s'effondre a **61.7%** d'accuracy. Cela met en evidence la difficulte pour un reseau de neurones simple d'apprendre des filtres visuels complexes a partir de zero avec si peu d'exemples d'entrainement.

**Interpretation graphique (Courbe ROC du SVM via InceptionV3)** :
Malgre les 9 classes, la methode "One-vs-Rest" affiche des courbes ROC excellentes pour chaque animal (AUC > 0.99), prouvant que l'espace des descripteurs d'InceptionV3 separe naturellement les especes animales.
![Courbe ROC SVM InceptionV3 DTD](reports/roc_SVM_via_InceptionV3_DTD.png)

**Interpretation graphique (Matrice de Confusion du k-NN InceptionV3)** :
La diagonale est ici presque immaculee. Les seules petites erreurs proviennent des classes tres similaires visuellement (ex: pelages d'animaux confondus).
![Matrice de confusion kNN DTD](reports/confusion_k-NN_via_InceptionV3_DTD.png)

---

## 7. Synthese Visuelle de la Comparaison

Les deux graphiques ci-dessous permettent de comparer l'ensemble de nos resultats en un coup d'oeil.

### 7.1. Comparaison detaillee par Jeu de Donnees
Ce graphique montre le classement des modeles pour chacun des quatre datasets. Le CNN personnalise est mis en valeur en bleu electrique. On y voit clairement sa superiorite relative sur les gros datasets et ses faiblesses sur les petits, face aux algorithmes traditionnels ultra-stables.
![Comparaison des performances par Dataset](reports/comparatif_global_accuracy.png)

### 7.2. Classement Final Global
Ce graphique presente le classement moyen de chaque architecture sur l'ensemble des jeux de donnees. Le trio de tete est occupe par le SVM sur InceptionV3 et VGG16, confirmant que la synergie entre Deep Learning (pour l'extraction) et Machine Learning classique (pour la classification) reste la strategie la plus robuste.
![Classement Final Global Moyen](reports/classement_final_moyen.png)

---

## 8. Discussion Critique et Analyse des Compromis

### 8.1. Efficacite du Transfer Learning (Pipeline 1)
Le Transfer Learning (Pipeline 1) se revele etre l'approche la plus stable et la plus performante. Il surpasse le CNN personnalise sur tous les jeux de donnees, en particulier sur Iris et DTD ou le volume d'images est tres reduit. Le fait d'utiliser des extracteurs entraines sur ImageNet (plus d'un million d'images d'objets du quotidien) permet de beneficier d'un espace de representation universel tres riche que les classifieurs classiques n'ont plus qu'a decouper lineairement ou non.

### 8.2. Cout d'Apprentissage et Temps d'Entrainement
Les modeles traditionnels (SVM, k-NN) sur descripteurs pre-extraits s'entrainent en une fraction de seconde (de 0.5 a 20 secondes). Cependant, la phase initiale d'extraction (faire passer toutes les images dans VGG16 ou InceptionV3) est lourde et longue en CPU. Pour le CNN personnalise, le temps d'entrainement est le plus lourd (jusqu'a 11 minutes sur CPU pour 15 epoques), car il doit ajuster tous ses poids en meme temps a chaque iteration.

### 8.3. Taille des Modeles et Contraintes Memoires
Bien que le Pipeline 1 soit extremement precis, il impose de manipuler de tres gros modeles. Par exemple, **VGG16 necessite 138 millions de parametres** (ce qui represente un fichier de stockage de plus de 500 Mo). Notre **CNN personnalise, quant a lui, ne necessite que 8.7 millions de parametres** (environ 35 Mo sur le disque). Cela en fait une option de choix pour un deploiement sur des systemes embarques legers (drones, micro-ordinateurs) ou la memoire vive est limitee.

---

## 9. Conclusion et Recommandations d'Usage

En reponse a notre problematique de depart, nous formulons les recommandations suivantes :
- **Recommandation 1** : Si la base d'images d'entrainement est petite (moins de 1000 images) et que les ressources de calcul sont limitees, le **Pipeline 1 (InceptionV3 ou VGG16 + SVM/k-NN)** est l'approche ideale. Elle assure une precision maximale et un temps d'entrainement presque instantane.
- **Recommandation 2** : Si le volume d'images est important et que la taille memoire finale du modele est un facteur limitant (par exemple pour du deploiement embarque), le **Pipeline 2 (CNN personnalise)** est a privilegier. Bien que son entrainement soit plus long a converger, il permet d'obtenir un modele tres compact, directement specifique aux structures de vos images de travail.
