import os
import torch

# on fixe une graine aleatoire pour que les resultats soient les memes a chaque execution
SEED = 42

# on detecte automatiquement si une carte graphique est disponible, sinon on utilise le CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# chemin vers le dossier Datasets qui contient toutes nos images
BASE_DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Datasets"))

# chemins complets vers chaque jeu de donnees
DATASET_PATHS = {
    "Covid19-XRAYS": os.path.join(BASE_DATASETS_DIR, "Covid19-XRAYS", "Covid19-XRAYS"),
    "DTD":           os.path.join(BASE_DATASETS_DIR, "DTD", "DTD"),
    "Iris":          os.path.join(BASE_DATASETS_DIR, "Iris"),
    "Wildfire":      os.path.join(BASE_DATASETS_DIR, "Wildfire")
}

# liste des classes valides pour chaque dataset
# pour Iris par exemple, on exclut le sous-dossier INF5082 qui n'est pas une classe de fleur
DATASET_CLASSES = {
    "Covid19-XRAYS": ["CT_COVID", "CT_NonCOVID"],
    "DTD":           ["antelope", "badger", "butterfly", "cat", "chimpanzee", "cow", "dragonfly", "eagle", "elephant"],
    "Iris":          ["iris-setosa", "iris-versicolour", "iris-virginica"],
    "Wildfire":      ["fire", "nofire"]
}

# tailles d'image pour chaque type de modele
TAILLE_IMAGE_VGG_ALEX = 224   # VGG16 et AlexNet ont besoin de 224x224
TAILLE_IMAGE_INCEPTION = 299  # InceptionV3 a besoin de 299x299
TAILLE_IMAGE_CNN = 128        # on prend 128x128 pour notre CNN, c'est un bon compromis vitesse/qualite

# dossiers de sortie du projet
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_DIR = os.path.join(BASE_PROJECT_DIR, "features")
MODELS_DIR   = os.path.join(BASE_PROJECT_DIR, "models")
REPORTS_DIR  = os.path.join(BASE_PROJECT_DIR, "reports")

# on cree les dossiers s'ils n'existent pas encore
for dossier in [FEATURES_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(dossier, exist_ok=True)

# hyperparametres pour l'entrainement du CNN personnalise
BATCH_SIZE    = 32
EPOCHS        = 15
LEARNING_RATE = 0.001
PROPORTION_TEST = 0.3  # 70% entrainement, 30% test, comme demande dans l'enonce
