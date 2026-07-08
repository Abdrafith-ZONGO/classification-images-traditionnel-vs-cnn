import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.config import DEVICE, MODELS_DIR, LEARNING_RATE, EPOCHS

# --- Définition de l'architecture du CNN ---
class CNNPerso(nn.Module):
    """
    Réseau de neurones convolutif personnalisé pour la classification d'images.
    
    Architecture (4 couches de convolution) :
    - Conv1 (3 -> 32 filtres, 3x3) -> BatchNorm -> ReLU -> MaxPool (2x2)
    - Conv2 (32 -> 64 filtres, 3x3) -> BatchNorm -> ReLU -> MaxPool (2x2)
    - Conv3 (64 -> 128 filtres, 3x3) -> BatchNorm -> ReLU -> MaxPool (2x2)
    - Conv4 (128 -> 256 filtres, 3x3) -> BatchNorm -> ReLU -> MaxPool (2x2)
    - Fully Connected (FC) : Flatten -> Linéaire(512) -> BatchNorm -> ReLU -> Dropout(0.5) -> Linéaire(nb_classes)
    
    Justification de l'architecture : L'augmentation progressive du nombre de filtres (32 -> 256)
    permet d'extraire des caractéristiques de plus en plus abstraites (des bords géométriques
    jusqu'aux motifs complexes). L'utilisation de BatchNorm accélère et stabilise la convergence,
    et le Dropout à 50% sur la couche dense limite fortement le risque de surapprentissage.
    """
    def __init__(self, nb_classes):
        super(CNNPerso, self).__init__()
        
        # Bloc 1 (Entrée: 128x128x3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Bloc 2 (Entrée: 64x64x32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Bloc 3 (Entrée: 32x32x64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Bloc 4 (Entrée: 16x16x128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Poolings et Activations
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Classifieur (Taille après 4 poolings de facteur 2 : 128 / 16 = 8. Donc map de 8x8x256)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, nb_classes)
        
    def forward(self, x):
        # Passage dans les blocs de convolution
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Aplatir les caractéristiques spatiales
        x = x.view(x.size(0), -1)
        
        # Passage dans les couches denses (FC)
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# --- Boucle d'entraînement et d'évaluation ---
def entrainer_evaluer_cnn(nom_dataset, loader_train, loader_test, nb_classes, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Entraîne le CNN personnalisé sur les images brutes et l'évalue sur le jeu de test.
    Mesure le temps d'entraînement et compte le nombre de paramètres du modèle.
    """
    print(f"Début de l'entraînement du CNN personnalisé sur {nom_dataset} ({epochs} époques)...")
    
    # 1. Instanciation du modèle, de la loss et de l'optimiseur
    modele = CNNPerso(nb_classes).to(DEVICE)
    critere = nn.CrossEntropyLoss()
    optimiseur = optim.Adam(modele.parameters(), lr=lr)
    
    # 2. Calcul du nombre de paramètres du modèle (critère de comparaison demandé)
    nb_parametres = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    print(f"   -> Nombre de paramètres du CNN : {nb_parametres:,}")
    
    historique = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    debut_temps = time.time()
    
    # 3. Boucle d'entraînement époque par époque
    for epoque in range(epochs):
        modele.train()
        perte_train = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in loader_train:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            sorties = modele(images)
            perte = critere(sorties, labels)
            
            # Rétropropagation
            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()
            
            # Métriques d'entraînement
            perte_train += perte.item() * images.size(0)
            _, predits = torch.max(sorties, 1)
            total_train += labels.size(0)
            correct_train += (predits == labels).sum().item()
            
        acc_train = correct_train / total_train
        perte_train_moy = perte_train / len(loader_train.dataset)
        
        # Évaluation rapide à chaque époque pour le suivi de la courbe d'apprentissage
        modele.eval()
        perte_test = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in loader_test:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                sorties = modele(images)
                perte = critere(sorties, labels)
                
                perte_test += perte.item() * images.size(0)
                _, predits = torch.max(sorties, 1)
                total_test += labels.size(0)
                correct_test += (predits == labels).sum().item()
                
        acc_test = correct_test / total_test
        perte_test_moy = perte_test / len(loader_test.dataset)
        
        historique['train_loss'].append(perte_train_moy)
        historique['train_acc'].append(acc_train)
        historique['test_loss'].append(perte_test_moy)
        historique['test_acc'].append(acc_test)
        
        # Affichage régulier de la progression de l'entraînement
        if (epoque + 1) % 5 == 0 or epoque == 0 or epoque == epochs - 1:
            print(f"   Époque [{epoque+1}/{epochs}] - Perte Train: {perte_train_moy:.4f} | Acc Train: {acc_train*100:.1f}% - Perte Test: {perte_test_moy:.4f} | Acc Test: {acc_test*100:.1f}%")
            
    temps_entrainement = time.time() - debut_temps
    print(f"   -> Entraînement terminé en {temps_entrainement:.2f} secondes")
    
    # 4. Évaluation finale pour collecter les prédictions et probabilités de test
    modele.eval()
    y_vrai = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for images, labels in loader_test:
            images = images.to(DEVICE)
            sorties = modele(images)
            probabilites = torch.softmax(sorties, dim=1)
            _, predits = torch.max(sorties, 1)
            
            y_vrai.extend(labels.numpy())
            y_pred.extend(predits.cpu().numpy())
            y_prob.extend(probabilites.cpu().numpy())
            
    y_vrai = np.array(y_vrai)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # 5. Sauvegarde du modèle PyTorch (.pth)
    chemin_modele = os.path.join(MODELS_DIR, f"model_cnn_{nom_dataset}.pth")
    torch.save(modele.state_dict(), chemin_modele)
    print(f"   -> Poids du CNN sauvegardés dans {chemin_modele}")
    
    # Création du dictionnaire récapitulatif
    resultats = {
        'historique': historique,
        'y_vrai': y_vrai,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'temps_entrainement': temps_entrainement,
        'nb_parametres': nb_parametres
    }
    
    return resultats, modele
