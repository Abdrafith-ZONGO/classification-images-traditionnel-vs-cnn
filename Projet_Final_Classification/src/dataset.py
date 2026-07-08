import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from src.config import (
    SEED, DATASET_PATHS, DATASET_CLASSES, BATCH_SIZE, PROPORTION_TEST
)

# --- Transformations des images ---
# Ces transformations normalisent les images pour les modèles pré-entraînés (ImageNet)
def obtenir_transforms(taille_image, augmentation=False):
    """
    Retourne les transformations PyTorch pour les images.
    Si augmentation=True, applique de la data augmentation pour l'entraînement.
    """
    # Moyenne et écart-type standard pour ImageNet (requis pour VGG, AlexNet, Inception)
    normalisation = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augmentation:
        # Augmentation pour régulariser et éviter le surapprentissage (CNN perso)
        return transforms.Compose([
            transforms.Resize((taille_image, taille_image)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalisation
        ])
    else:
        # Transformation de base pour l'évaluation et l'extraction
        return transforms.Compose([
            transforms.Resize((taille_image, taille_image)),
            transforms.ToTensor(),
            normalisation
        ])


# --- Dataset personnalisé PyTorch ---
class ImageDatasetCustom(Dataset):
    """
    Classe de Dataset PyTorch chargée de lire les images sur le disque.
    Elle est découplée de la structure physique grâce au passage direct des chemins.
    """
    def __init__(self, chemins_images, labels, transform=None):
        self.chemins_images = chemins_images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.chemins_images)

    def __getitem__(self, idx):
        chemin = self.chemins_images[idx]
        label = self.labels[idx]
        
        # Ouverture de l'image en s'assurant qu'elle est en mode RGB (3 canaux)
        image = Image.open(chemin).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


# --- Fonction principale de chargement et de filtrage ---
def collecter_donnees_dataset(nom_dataset):
    """
    Explore le dossier d'un dataset, valide les fichiers, exclut les répertoires parasites
    comme 'INF5082' de façon logique (programmation), et renvoie les listes de chemins et labels.
    
    Retourne:
        chemins_valides (list): Chemins absolus des images.
        labels_valides (list): Indices numériques des classes.
        classes (list): Noms des classes valides associées.
    """
    chemin_racine = DATASET_PATHS[nom_dataset]
    classes_autorisees = DATASET_CLASSES[nom_dataset]
    
    chemins_valides = []
    labels_valides = []
    
    # Création d'une correspondance nom_classe -> index
    classe_vers_index = {nom: idx for idx, nom in enumerate(classes_autorisees)}
    
    # Extensions d'images valides
    extensions_valides = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    # Parcours des classes autorisées uniquement (gestion propre de INF5082 pour Iris)
    for nom_classe in classes_autorisees:
        dossier_classe = os.path.join(chemin_racine, nom_classe)
        if not os.path.isdir(dossier_classe):
            continue
            
        index_classe = classe_vers_index[nom_classe]
        
        # Analyse des fichiers dans ce dossier
        for fichier in os.listdir(dossier_classe):
            extension = os.path.splitext(fichier)[1].lower()
            if extension in extensions_valides:
                chemin_complet = os.path.join(dossier_classe, fichier)
                
                # Vérification de l'intégrité de l'image
                try:
                    with Image.open(chemin_complet) as img:
                        img.verify() # Vérifie si l'image n'est pas corrompue
                    chemins_valides.append(chemin_complet)
                    labels_valides.append(index_classe)
                except Exception:
                    # En cas d'erreur de lecture, l'image est ignorée silencieusement pour le pipeline
                    pass
                    
    return chemins_valides, labels_valides, classes_autorisees


def preparer_loaders(nom_dataset, taille_image, batch_size=BATCH_SIZE, test_size=PROPORTION_TEST):
    """
    Prépare et retourne les DataLoader d'entraînement et de test avec un split stratifié.
    
    Justification méthodologique : le split stratifié (via sklearn) garantit une répartition
    équitable des classes entre l'ensemble d'entraînement et de test, évitant ainsi le déséquilibre.
    """
    chemins, labels, classes = collecter_donnees_dataset(nom_dataset)
    
    if len(chemins) == 0:
        raise ValueError(f"Aucune image valide trouvée pour le dataset {nom_dataset}")
        
    # Split stratifié (70% Train, 30% Test)
    chemins_train, chemins_test, labels_train, labels_test = train_test_split(
        chemins, labels,
        test_size=test_size,
        stratify=labels,
        random_state=SEED
    )
    
    # Création des objets Dataset avec et sans data augmentation
    dataset_train = ImageDatasetCustom(
        chemins_train, labels_train,
        transform=obtenir_transforms(taille_image, augmentation=True)
    )
    
    dataset_test = ImageDatasetCustom(
        chemins_test, labels_test,
        transform=obtenir_transforms(taille_image, augmentation=False)
    )
    
    # Création des DataLoaders pour l'entraînement (shuffle=True) et le test (shuffle=False)
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    loader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return loader_train, loader_test, classes
