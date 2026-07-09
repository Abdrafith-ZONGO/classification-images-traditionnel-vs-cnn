import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from src.config import (
    SEED, DATASET_PATHS, DATASET_CLASSES, BATCH_SIZE, PROPORTION_TEST
)


def obtenir_transforms(taille_image, augmentation=False):
    """
    Retourne les transformations a appliquer sur les images avant de les passer au modele.
    Si augmentation=True, on ajoute des transformations aleatoires pour eviter le surapprentissage.
    Si augmentation=False, on fait juste le redimensionnement et la normalisation standard.
    """
    # valeurs de normalisation recommandees pour tous les modeles pre-entraines sur ImageNet
    normalisation = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if augmentation:
        # pour l'entrainement du CNN, on augmente les donnees pour regulariser
        # retournement horizontal, petite rotation et legere variation de couleur
        return transforms.Compose([
            transforms.Resize((taille_image, taille_image)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalisation
        ])
    else:
        # pour l'extraction de features et l'evaluation, pas de modification aleatoire
        return transforms.Compose([
            transforms.Resize((taille_image, taille_image)),
            transforms.ToTensor(),
            normalisation
        ])


class ImageDatasetCustom(Dataset):
    """
    Notre propre classe de Dataset PyTorch.
    Elle charge les images depuis le disque a partir d'une liste de chemins.
    On lui passe directement les chemins et les labels numeriques correspondants.
    """

    def __init__(self, chemins_images, labels, transform=None):
        self.chemins_images = chemins_images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.chemins_images)

    def __getitem__(self, idx):
        chemin = self.chemins_images[idx]
        label  = self.labels[idx]

        # on ouvre l'image et on la convertit en RGB pour avoir 3 canaux (meme pour les images en gris)
        image = Image.open(chemin).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def collecter_donnees_dataset(nom_dataset):
    """
    Parcourt le dossier d'un dataset et retourne les chemins des images valides avec leurs labels.
    On filtre uniquement les classes qu'on a definies dans config.py.
    On verifie aussi que chaque image n'est pas corrompue avant de l'ajouter a la liste.
    """
    chemin_racine    = DATASET_PATHS[nom_dataset]
    classes_autorisees = DATASET_CLASSES[nom_dataset]

    chemins_valides = []
    labels_valides  = []

    # on associe chaque nom de classe a un numero (ex: "iris-setosa" -> 0)
    classe_vers_index = {nom: idx for idx, nom in enumerate(classes_autorisees)}

    # extensions d'images qu'on accepte
    extensions_valides = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for nom_classe in classes_autorisees:
        dossier_classe = os.path.join(chemin_racine, nom_classe)

        # si le dossier n'existe pas, on passe a la classe suivante
        if not os.path.isdir(dossier_classe):
            continue

        index_classe = classe_vers_index[nom_classe]

        for fichier in os.listdir(dossier_classe):
            extension = os.path.splitext(fichier)[1].lower()

            if extension in extensions_valides:
                chemin_complet = os.path.join(dossier_classe, fichier)

                # on verifie que l'image peut etre ouverte correctement
                try:
                    with Image.open(chemin_complet) as img:
                        img.verify()
                    chemins_valides.append(chemin_complet)
                    labels_valides.append(index_classe)
                except Exception:
                    # image corrompue, on l'ignore
                    pass

    return chemins_valides, labels_valides, classes_autorisees


def preparer_loaders(nom_dataset, taille_image, batch_size=BATCH_SIZE, test_size=PROPORTION_TEST):
    """
    Charge un dataset et retourne deux DataLoaders : un pour l'entrainement, un pour le test.
    On fait un split stratifie pour que la proportion des classes soit respectee dans les deux ensembles.
    Le split est 70% train et 30% test comme demande dans l'enonce.
    """
    chemins, labels, classes = collecter_donnees_dataset(nom_dataset)

    if len(chemins) == 0:
        raise ValueError(f"Aucune image valide trouvee pour le dataset {nom_dataset}")

    # split stratifie : on garde la meme proportion de chaque classe dans train et test
    chemins_train, chemins_test, labels_train, labels_test = train_test_split(
        chemins, labels,
        test_size=test_size,
        stratify=labels,
        random_state=SEED
    )

    # dataset d'entrainement avec augmentation de donnees
    dataset_train = ImageDatasetCustom(
        chemins_train, labels_train,
        transform=obtenir_transforms(taille_image, augmentation=True)
    )

    # dataset de test sans augmentation (on veut evaluer sur des images normales)
    dataset_test = ImageDatasetCustom(
        chemins_test, labels_test,
        transform=obtenir_transforms(taille_image, augmentation=False)
    )

    # on melange les donnees d'entrainement a chaque epoque, mais pas le test
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,  num_workers=0)
    loader_test  = DataLoader(dataset_test,  batch_size=batch_size, shuffle=False, num_workers=0)

    return loader_train, loader_test, classes
