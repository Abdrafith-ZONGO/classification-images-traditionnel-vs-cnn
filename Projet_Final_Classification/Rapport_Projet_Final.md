# Rapport de Projet - Classification d'images par Apprentissage Automatique Traditionnel et Reseaux de Neurones Convolutifs (CNN)

**Dépôt GitHub** : [classification-images-traditionnel-vs-cnn](https://github.com/Abdrafith-ZONGO/classification-images-traditionnel-vs-cnn.git)  
**Module** : Reseaux de Neurones Artificiels  
**Niveau** : Master 1 - Vision par Ordinateur & Intelligence Artificielle  

---

## 1. Introduction
La classification d'images constitue l'un des piliers fondamentaux de la vision par ordinateur moderne, avec des applications cles allant de l'imagerie medicale à l'analyse d'images satellites et à la robotique. Historiquement, deux grandes approches se sont succede pour resoudre cette tache :
1. **L'approche traditionnelle** (Machine Learning classique) : elle repose sur une separation stricte entre l'extraction manuelle ou semi-manuelle de caracteristiques (features) et la classification (par exemple via des SVM ou des forets aleatoires).
2. **L'approche par apprentissage profond** (Deep Learning) : elle utilise des reseaux convolutifs (CNN) capables d'apprendre conjointement l'extraction de caracteristiques spatiales et la classification directement à partir des pixels bruts.

Dans ce projet, nous mettons en oeuvre et comparons ces deux methodes à l'aide de deux pipelines distincts executes sur quatre jeux de données : Iris Flowers (images), COVID-19 X-Ray, Wildfire Satellite Images, et DTD (Describable Textures Dataset).

---

## 2. Problematique
L'apprentissage profond a demontre une superiorite ecrasante sur les grands jeux de donnees, mais il necessite des ressources de calcul massives (GPU) et des volumes de donnees consequents pour eviter le surapprentissage (overfitting). À l'inverse, le Transfer Learning (utilisation de modeles pre-entraines comme extracteurs) combine aux classifieurs traditionnels offre une alternative legere et rapide. 

Ce travail vise à repondre aux questions suivantes :
- Quels sont les compromis exactitude / temps de calcul entre un CNN entraine a partir de zero et des classifieurs classiques sur descripteurs pre-entraines ?
- Quelle approche est la plus viable selon la taille du dataset et le contexte materiel (notamment lors de l'utilisation d'un CPU) ?

---

## 3. Methodologie et Architecture des Pipelines

### 3.1. Preparation et Pretraitement des Donnees
Pour chaque jeu de donnees, nous mettons en place un protocole rigoureux de pretraitement :
- **Filtres de validation** : Les images sont verifiees programmatiquement via la bibliotheque PIL afin d'exclure les fichiers corrompus avant l'apprentissage.
- **Filtrage des classes** : Pour le dataset Iris, le repertoire parasite "INF5082" (TP tiers) est filtre au niveau du code de chargement pour ne conserver que les classes valides : `iris-setosa`, `iris-versicolour` et `iris-virginica`.
- **Decoupage train/test** : Les donnees sont divisees en 70% pour l'entrainement et 30% pour le test. Nous appliquons un **decoupage stratifie** (stratified split) pour maintenir la distribution d'origine des classes dans chaque sous-ensemble.
- **Normalisation** : Les images destinees aux modeles pre-entraines sont redimensionnees (224x224 pour VGG16/AlexNet, 299x299 pour InceptionV3) et normalisees avec les moyennes et ecarts-types de la base ImageNet. Les images pour le CNN perso sont redimensionnees en 128x128.

### 3.2. Pipeline 1 : Extraction de Features et Classifieurs Traditionnels
Nous utilisons trois architectures convolutives de reference pre-entrainees sur ImageNet comme extracteurs de caracteristiques :
- **AlexNet** : features extraites avant la derniere couche de classification (taille du vecteur : 4096).
- **VGG16** : features extraites de l'avant-derniere couche dense (taille du vecteur : 4096).
- **InceptionV3** : features extraites apres la couche de pooling globale (taille du vecteur : 2048).

Une fois les features extraites et stockees (au format `.npy`), nous entrainons quatre classifieurs apres une etape de normalisation standard (`StandardScaler`) :
- **SVM** : noyau RBF ou Lineaire, avec optimisation par grille de recherche (`GridSearchCV`) sur le parametre de regularisation $C \in \{0.1, 1, 10\}$.
- **k-NN** : optimisation du nombre de voisins $k \in \{3, 5, 7, 11\}$ et de la metrique (Euclidienne vs Manhattan).
- **Arbre de Decision** : controle de la profondeur (`max_depth` $\in \{3, 5, 10, None\}$) pour eviter le surapprentissage.
- **Naïve Bayes** : classifieur Gaussien utilise comme baseline probabiliste.

### 3.3. Pipeline 2 : CNN Personnalise
L'architecture de notre CNN personnalise (code dans `src/models_cnn.py`) est composee de quatre blocs convolutifs successifs conçus pour limiter le nombre de parametres tout en preservant le pouvoir d'abstraction spatiale :
1. **Bloc 1** : Conv2D (3 vers 32 filtres, 3x3) + Batch Normalization + ReLU + MaxPool (2x2)
2. **Bloc 2** : Conv2D (32 vers 64 filtres, 3x3) + Batch Normalization + ReLU + MaxPool (2x2)
3. **Bloc 3** : Conv2D (64 vers 128 filtres, 3x3) + Batch Normalization + ReLU + MaxPool (2x2)
4. **Bloc 4** : Conv2D (128 vers 256 filtres, 3x3) + Batch Normalization + ReLU + MaxPool (2x2)
5. **Couche Fully Connected** : Aplatissement (Flatten) + Lineaire (512 neurones) + Batch Normalization + ReLU + Dropout (50%) + Lineaire de sortie (taille egale au nombre de classes).

L'optimisation est realisee avec l'algorithme Adam (Taux d'apprentissage = 0.001) et la fonction de perte CrossEntropyLoss.

---

## 4. Resultats Experimentaux et Analyses Comparatives
Les resultats suivants representent les performances obtenues sur l'ensemble des datasets a l'issue de l'execution du pipeline complet.

### 4.1. Analyse Globale des Resultats
Le tableau ci-dessous regroupe les metriques cles de performance pour chaque dataset et modele :

| Dataset | Pipeline | Modele | Accuracy | Precision | Recall | F1-Score | Temps Entr. (s) | Parametres |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Iris** | Pipeline 1 | SVM (via InceptionV3) | **0.6614** | 0.6832 | 0.6614 | 0.5878 | 21.15 | 27,161,264 |
| | Pipeline 1 | k-NN (via VGG16) | **0.6693** | 0.7822 | 0.6693 | 0.5638 | 2.28 | 138,357,544 |
| | Pipeline 2 | CNN Personnalise | 0.6378 | 0.5963 | 0.6378 | 0.5890 | 249.01 | 8,781,059 |
| **Covid19** | Pipeline 1 | SVM (via InceptionV3) | **0.8251** | 0.8271 | 0.8251 | 0.8253 | 55.11 | 27,161,264 |
| | Pipeline 1 | Naive Bayes (via InceptionV3) | 0.7623 | 0.7691 | 0.7623 | 0.7586 | 0.46 | 27,161,264 |
| | Pipeline 2 | CNN Personnalise | 0.7668 | 0.7901 | 0.7668 | 0.7586 | 326.65 | 8,780,546 |
| **Wildfire**| Pipeline 1 | SVM (via AlexNet) | **0.9927** | 0.9928 | 0.9927 | 0.9927 | 321.12 | 61,100,840 |
| | Pipeline 1 | k-NN (via VGG16) | **0.9945** | 0.9946 | 0.9945 | 0.9945 | 20.82 | 138,357,544 |
| | Pipeline 2 | CNN Personnalise | 0.9527 | 0.9543 | 0.9527 | 0.9527 | 676.31 | 8,780,546 |
| **DTD** | Pipeline 1 | SVM (via InceptionV3) | **0.9877** | 0.9883 | 0.9877 | 0.9876 | 22.49 | 27,161,264 |
| | Pipeline 1 | k-NN (via InceptionV3) | **0.9877** | 0.9883 | 0.9877 | 0.9876 | 3.20 | 27,161,264 |
| | Pipeline 2 | CNN Personnalise | 0.6173 | 0.6532 | 0.6173 | 0.6200 | 323.09 | 8,784,137 |

### 4.2. Analyse par Dataset

#### 4.2.1. Dataset Iris Flowers
Le dataset Iris est de petite taille (421 images au total). L'approche traditionnelle se montre tres efficace, notamment l'association d'extracteurs comme **VGG16** avec un classifieur simple comme le **k-NN** (Accuracy = **66.93%**). Les classifieurs classiques tirent profit des representions robustes pre-entrainees. Le CNN personnalise, bien que performant (**63.78%**), a tendance a legerement surapprendre en raison du faible volume de donnees disponible par classe.

#### 4.2.2. Dataset COVID-19 X-Ray
Sur ce dataset de 743 images de radiographies CT-scan, le pipeline traditionnel avec l'extracteur **InceptionV3** associe au **SVM (GridSearchCV)** obtient le score le plus eleve de **82.51%** d'accuracy et un F1-score de **0.8253**. Le CNN personnalise se defend tres bien avec **76.68%** d'accuracy, demontrant que la structure convolutive a appris des filtres pertinents pour detecter les opacites pulmonaires typiques du COVID-19.

#### 4.2.3. Dataset Wildfire Satellite Images
Ce dataset contient 1 832 images satellites de zones incendiees vs non incendiees. Le signal visuel etant tres tranche, toutes les methodes obtiennent des scores exceptionnels. Le **k-NN sur features VGG16** atteint **99.45%** d'accuracy de test. Le CNN personnalise obtient quant a lui **95.27%** d'accuracy, ce qui represente une performance solide pour un modele entraine a partir de zero en quelques minutes.

#### 4.2.4. Dataset DTD (Textures et Animaux)
Ce dataset contient 540 images divisees en 9 categories d'animaux. L'extracteur **InceptionV3** combine a un **SVM** ou a un **k-NN** surclasse totalement les autres approches avec **98.77%** d'accuracy. Le CNN personnalise plafonne a **61.73%**, illustrant la difficulte pour un reseau de neurones convolutif simple d'apprendre des filtres robustes pour 9 classes complexes avec seulement 60 images par classe d'entrainement.

---

## 5. Discussion Critique

1. **Efficacite du Transfer Learning** :
   Le Transfer Learning (Pipeline 1) s'est montre systematiquement superieur en termes de precision pure et de stabilite sur les quatre bases de donnees, en particulier sur les bases a faible echantillonnage (DTD, Iris). L'utilisation de modeles pre-entraines sur plus d'un million d'images d'ImageNet fournit un espace representationnel tres riche qu'un classifieur classique peut aisement separer.

2. **Cout d'apprentissage et Temps d'Entrainement** :
   Les classifieurs traditionnels (SVM, k-NN) sur les features pre-extraites s'entrainent en une fraction de seconde (de 0.5 à 20 secondes). Cependant, le passage des images dans le grand modele (AlexNet ou VGG16) necessite un temps de calcul initial important. Pour le CNN personnalisé, le temps de calcul est proportionnellement eleve lors de l'apprentissage (jusqu'a 11 minutes pour 10 epoques sur Wildfire avec CPU), mais le modele final est extremement leger.

3. **Complexite de l'architecture et memoire** :
   VGG16 necessite 138 millions de parametres, ce qui correspond a un fichier de stockage lourd et volumineux. Le CNN personnalise utilise seulement **8.7 millions de parametres**, offrant un avantage considerable si le modele doit etre deploye sur des systemes embarques avec des contraintes memoires strictes.

---

## 6. Conclusion et Recommandations
- **Recommandation 1** : Pour des applications ayant des ressources de calcul moderees (par exemple sans GPU) et de faibles volumes de donnees d'entrainement (ex: Iris ou DTD), le **Pipeline 1 (Extraction InceptionV3 ou VGG16 + SVM/KNN)** est l'approche a privilegier. Elle combine rapidite de developpement, precision maximale et stabilite du resultat.
- **Recommandation 2** : Pour des bases de donnees volumineuses ou des environnements de deploiement ayant de fortes contraintes memoires (ex: embarque), le **Pipeline 2 (CNN personnalisé)** est a preferer, car il permet de generer un modele compact directement adapte aux specificites des images brutes.
