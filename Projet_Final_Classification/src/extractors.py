import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from src.config import DEVICE, FEATURES_DIR, TAILLE_IMAGE_VGG_ALEX, TAILLE_IMAGE_INCEPTION
from src.dataset import preparer_loaders

def charger_modele_preentraine(nom_modele):
    """
    Charge un modèle pré-entraîné sur ImageNet et remplace la dernière couche de classification
    par une identité (nn.Identity) pour l'utiliser comme extracteur de caractéristiques.
    
    Retourne:
        modele (nn.Module): Le modèle modifié.
        dimension_sortie (int): La taille du vecteur de caractéristiques extrait.
    """
    print(f"Chargement du modèle pré-entraîné : {nom_modele}...")
    
    if nom_modele == "AlexNet":
        modele = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        dimension_sortie = modele.classifier[6].in_features  # 4096 features
        modele.classifier[6] = nn.Identity()                 # Suppression de la couche finale
        
    elif nom_modele == "VGG16":
        modele = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        dimension_sortie = modele.classifier[6].in_features  # 4096 features
        modele.classifier[6] = nn.Identity()
        
    elif nom_modele == "InceptionV3":
        # InceptionV3 requiert aux_logits=False si on ne veut pas de sorties auxiliaires en eval
        modele = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        modele.aux_logits = False
        dimension_sortie = modele.fc.in_features            # 2048 features
        modele.fc = nn.Identity()
        
    else:
        raise ValueError(f"Modèle inconnu : {nom_modele}")
        
    modele = modele.to(DEVICE)
    modele.eval() # Désactive le Dropout et la BatchNorm d'entraînement
    return modele, dimension_sortie


def extraire_features(modele, loader):
    """
    Passe toutes les images d'un DataLoader dans le modèle pour en extraire
    les représentations vectorielles (features) et les labels associés.
    """
    toutes_features = []
    tous_labels = []
    
    # Pas de calcul de gradient requis pour l'extraction de features
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extraction", leave=False):
            images = images.to(DEVICE)
            
            # Passage dans le modèle
            sorties = modele(images)
            
            # Aplatir le vecteur si nécessaire (pour Inception ou VGG)
            sorties = sorties.view(sorties.size(0), -1)
            
            toutes_features.append(sorties.cpu().numpy())
            tous_labels.append(labels.numpy())
            
    # Empilement des vecteurs extraits pour obtenir une grande matrice numpy
    features_concat = np.vstack(toutes_features)
    labels_concat = np.concatenate(tous_labels)
    
    return features_concat, labels_concat


def obtenir_features_dataset(nom_dataset, nom_modele, forcer_reextraction=False):
    """
    Gère l'extraction complète pour un dataset et un modèle donné.
    Sauvegarde et charge depuis les fichiers .npy pour éviter de recalculer inutilement.
    
    Justification méthodologique : La sauvegarde sous format binaire NumPy (.npy) permet de
    ne faire l'extraction (très coûteuse en temps CPU/GPU) qu'une seule fois.
    """
    chemin_train_feat = os.path.join(FEATURES_DIR, f"feat_train_{nom_dataset}_{nom_modele}.npy")
    chemin_train_lab = os.path.join(FEATURES_DIR, f"labels_train_{nom_dataset}_{nom_modele}.npy")
    chemin_test_feat = os.path.join(FEATURES_DIR, f"feat_test_{nom_dataset}_{nom_modele}.npy")
    chemin_test_lab = os.path.join(FEATURES_DIR, f"labels_test_{nom_dataset}_{nom_modele}.npy")
    
    # Vérification si les fichiers existent déjà pour éviter le calcul
    if not forcer_reextraction and all(os.path.exists(p) for p in [chemin_train_feat, chemin_train_lab, chemin_test_feat, chemin_test_lab]):
        print(f"Chargement des caractéristiques sauvegardées pour {nom_dataset} via {nom_modele}...")
        X_train = np.load(chemin_train_feat)
        y_train = np.load(chemin_train_lab)
        X_test = np.load(chemin_test_feat)
        y_test = np.load(chemin_test_lab)
        return X_train, y_train, X_test, y_test
        
    print(f"Extraction des caractéristiques pour {nom_dataset} avec {nom_modele}...")
    
    # InceptionV3 requiert des images de taille 299x299
    taille_image = TAILLE_IMAGE_INCEPTION if nom_modele == "InceptionV3" else TAILLE_IMAGE_VGG_ALEX
    
    # Préparation des loaders (split 70% train / 30% test)
    loader_train, loader_test, classes = preparer_loaders(nom_dataset, taille_image)
    
    # Chargement du modèle
    modele, dim = charger_modele_preentraine(nom_modele)
    
    # Extraction effective
    print("   -> Traitement du jeu d'entraînement...")
    X_train, y_train = extraire_features(modele, loader_train)
    print("   -> Traitement du jeu de test...")
    X_test, y_test = extraire_features(modele, loader_test)
    
    # Sauvegarde sur disque
    np.save(chemin_train_feat, X_train)
    np.save(chemin_train_lab, y_train)
    np.save(chemin_test_feat, X_test)
    np.save(chemin_test_lab, y_test)
    
    print(f"Extraction terminée et sauvegardée dans {FEATURES_DIR}")
    return X_train, y_train, X_test, y_test
