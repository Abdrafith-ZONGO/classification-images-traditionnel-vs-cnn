import os
import torch

# Graine aléatoire pour assurer la reproductibilité des résultats
SEED = 42

# Détection automatique de la carte graphique (GPU) si disponible pour accélérer les calculs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chemins vers les dossiers des données brutes
# Ces chemins pointent vers le dossier parent "Datasets/" contenant nos images
BASE_DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Datasets"))

DATASET_PATHS = {
    "Covid19-XRAYS": os.path.join(BASE_DATASETS_DIR, "Covid19-XRAYS", "Covid19-XRAYS"),
    "DTD": os.path.join(BASE_DATASETS_DIR, "DTD", "DTD"),
    "Iris": os.path.join(BASE_DATASETS_DIR, "Iris"),
    "Wildfire": os.path.join(BASE_DATASETS_DIR, "Wildfire")
}

# Classes valides pour chaque dataset (permet d'exclure les dossiers parasites comme INF5082 pour Iris)
DATASET_CLASSES = {
    "Covid19-XRAYS": ["CT_COVID", "CT_NonCOVID"],
    "DTD": ["antelope", "badger", "butterfly", "cat", "chimpanzee", "cow", "dragonfly", "eagle", "elephant"],
    "Iris": ["iris-setosa", "iris-versicolour", "iris-virginica"],
    "Wildfire": ["fire", "nofire"]
}

# Paramètres généraux pour le traitement d'images
TAILLE_IMAGE_VGG_ALEX = 224      # Taille standard pour VGG16 et AlexNet (224x224)
TAILLE_IMAGE_INCEPTION = 299     # Taille spécifique requise par InceptionV3 (299x299)
TAILLE_IMAGE_CNN = 128           # Taille raisonnable pour notre CNN perso (compromis vitesse/performance sur machine locale)

# Chemins de sauvegarde pour notre projet
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_DIR = os.path.join(BASE_PROJECT_DIR, "features")
MODELS_DIR = os.path.join(BASE_PROJECT_DIR, "models")
REPORTS_DIR = os.path.join(BASE_PROJECT_DIR, "reports")

# Création automatique des dossiers de sortie s'ils n'existent pas
for dossier in [FEATURES_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(dossier, exist_ok=True)

# Hyperparamètres d'entraînement du CNN
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
PROPORTION_TEST = 0.3  # Split 70% train / 30% test demandé
