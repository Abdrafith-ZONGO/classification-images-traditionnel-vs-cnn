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
    on charge un modele deja entraine sur ImageNet.
    on enleve sa derniere couche de classification (on la remplace par une identite)
    pour pouvoir s'en servir comme extracteur de caracteristiques.
    """
    print(f"Chargement du modele pre-entraine : {nom_modele}...")
    
    if nom_modele == "AlexNet":
        modele = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        dimension_sortie = modele.classifier[6].in_features  # 4096 caracteristiques
        modele.classifier[6] = nn.Identity()                 # on court-circuite la derniere couche
        
    elif nom_modele == "VGG16":
        modele = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        dimension_sortie = modele.classifier[6].in_features  # 4096 caracteristiques
        modele.classifier[6] = nn.Identity()
        
    elif nom_modele == "InceptionV3":
        # InceptionV3 a besoin de aux_logits=False pour l'evaluation
        modele = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        modele.aux_logits = False
        dimension_sortie = modele.fc.in_features            # 2048 caracteristiques
        modele.fc = nn.Identity()
        
    else:
        raise ValueError(f"Modele non pris en charge : {nom_modele}")
        
    modele = modele.to(DEVICE)
    modele.eval() # on desactive le dropout et la batchnorm
    return modele, dimension_sortie


def extraire_features(modele, loader):
    """
    on fait passer toutes les images du loader dans le modele
    pour recuperer leurs representations vectorielles et les etiquettes.
    """
    toutes_features = []
    tous_labels = []
    
    # pas besoin de calculer les gradients pour de l'extraction
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extraction", leave=False):
            images = images.to(DEVICE)
            
            # on passe les images dans le modele
            sorties = modele(images)
            
            # on aplatit le vecteur au cas ou
            sorties = sorties.view(sorties.size(0), -1)
            
            toutes_features.append(sorties.cpu().numpy())
            tous_labels.append(labels.numpy())
            
    # on regroupe tout dans des grands tableaux numpy
    features_concat = np.vstack(toutes_features)
    labels_concat = np.concatenate(tous_labels)
    
    return features_concat, labels_concat


def obtenir_features_dataset(nom_dataset, nom_modele, forcer_reextraction=False):
    """
    on gere l'extraction complete pour un dataset et un modele specifique.
    on enregistre les resultats dans des fichiers .npy pour ne pas avoir a tout refaire
    si on relance le code.
    """
    chemin_train_feat = os.path.join(FEATURES_DIR, f"feat_train_{nom_dataset}_{nom_modele}.npy")
    chemin_train_lab = os.path.join(FEATURES_DIR, f"labels_train_{nom_dataset}_{nom_modele}.npy")
    chemin_test_feat = os.path.join(FEATURES_DIR, f"feat_test_{nom_dataset}_{nom_modele}.npy")
    chemin_test_lab = os.path.join(FEATURES_DIR, f"labels_test_{nom_dataset}_{nom_modele}.npy")
    
    # si les fichiers existent deja et qu'on ne force pas, on les recharge directement
    if not forcer_reextraction and all(os.path.exists(p) for p in [chemin_train_feat, chemin_train_lab, chemin_test_feat, chemin_test_lab]):
        print(f"Chargement des caracteristiques pre-extraites pour {nom_dataset} via {nom_modele}...")
        X_train = np.load(chemin_train_feat)
        y_train = np.load(chemin_train_lab)
        X_test = np.load(chemin_test_feat)
        y_test = np.load(chemin_test_lab)
        return X_train, y_train, X_test, y_test
        
    print(f"Extraction des caracteristiques pour {nom_dataset} avec {nom_modele}...")
    
    # InceptionV3 prend des images de 299x299, les autres 224x224
    taille_image = TAILLE_IMAGE_INCEPTION if nom_modele == "InceptionV3" else TAILLE_IMAGE_VGG_ALEX
    
    # on recupere les loaders (avec le split 70-30)
    loader_train, loader_test, classes = preparer_loaders(nom_dataset, taille_image)
    
    # on charge l'extracteur
    modele, dim = charger_modele_preentraine(nom_modele)
    
    # extraction sur le train et le test
    print("   - Traitement des images d'entrainement...")
    X_train, y_train = extraire_features(modele, loader_train)
    print("   - Traitement des images de test...")
    X_test, y_test = extraire_features(modele, loader_test)
    
    # on sauvegarde tout sur le disque
    np.save(chemin_train_feat, X_train)
    np.save(chemin_train_lab, y_train)
    np.save(chemin_test_feat, X_test)
    np.save(chemin_test_lab, y_test)
    
    print(f"Sauvegarde terminee dans le dossier : {FEATURES_DIR}")
    return X_train, y_train, X_test, y_test
