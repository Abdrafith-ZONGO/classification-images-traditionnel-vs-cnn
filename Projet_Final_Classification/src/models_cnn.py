import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.config import DEVICE, MODELS_DIR, LEARNING_RATE, EPOCHS

class CNNPerso(nn.Module):
    """
    notre architecture de cnn personnalise.
    elle est faite de 4 blocs de convolution successifs pour extraire les formes de l'image.
    puis de deux couches denses pour faire la classification finale.
    """
    def __init__(self, nb_classes):
        super(CNNPerso, self).__init__()
        
        # bloc 1 : convolution simple (entree 128x128x3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # bloc 2 : entree 64x64x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # bloc 3 : entree 32x32x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # bloc 4 : entree 16x16x128
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # pooling et activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # classifieur final
        # apres 4 poolings par 2, notre image de 128x128 est devenue 8x8 (128 / 2^4 = 8)
        # on a 256 filtres, donc le vecteur d'entree de la couche dense fait 256 * 8 * 8
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5) # dropout a 50% pour eviter d'apprendre par coeur
        self.fc2 = nn.Linear(512, nb_classes)
        
    def forward(self, x):
        # passage dans les couches de convolution
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # on aplatit la matrice en un long vecteur
        x = x.view(x.size(0), -1)
        
        # passage dans le classifieur
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def entrainer_evaluer_cnn(nom_dataset, loader_train, loader_test, nb_classes, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    on entraine notre cnn sur les images d'entrainement puis on l'evalue sur le test.
    on compte aussi son nombre de parametres pour pouvoir le comparer aux gros modeles.
    """
    print(f"Debut de l'entrainement du CNN sur {nom_dataset} ({epochs} epoques)...")
    
    modele = CNNPerso(nb_classes).to(DEVICE)
    critere = nn.CrossEntropyLoss()
    optimiseur = optim.Adam(modele.parameters(), lr=lr)
    
    # on calcule le nombre de parametres total
    nb_parametres = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    print(f"   - Nombre de parametres du CNN : {nb_parametres:,}")
    
    historique = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    debut_temps = time.time()
    
    # boucle epoque par epoque
    for epoque in range(epochs):
        modele.train()
        perte_train = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in loader_train:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # forward
            sorties = modele(images)
            perte = critere(sorties, labels)
            
            # backward
            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()
            
            perte_train += perte.item() * images.size(0)
            _, predits = torch.max(sorties, 1)
            total_train += labels.size(0)
            correct_train += (predits == labels).sum().item()
            
        acc_train = correct_train / total_train
        perte_train_moy = perte_train / len(loader_train.dataset)
        
        # evaluation a la fin de l'epoque pour tracer la courbe d'apprentissage
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
        
        # affichage des performances toutes les 5 epoques
        if (epoque + 1) % 5 == 0 or epoque == 0 or epoque == epochs - 1:
            print(f"   Epoque [{epoque+1}/{epochs}] - Loss Train: {perte_train_moy:.4f} | Acc Train: {acc_train*100:.1f}% - Loss Test: {perte_test_moy:.4f} | Acc Test: {acc_test*100:.1f}%")
            
    temps_entrainement = time.time() - debut_temps
    print(f"   - Entrainement fini en : {temps_entrainement:.2f} secondes")
    
    # evaluation finale pour recuperer predictions et probabilites
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
    
    # on sauvegarde les poids du cnn sur le disque
    chemin_modele = os.path.join(MODELS_DIR, f"model_cnn_{nom_dataset}.pth")
    torch.save(modele.state_dict(), chemin_modele)
    print(f"   - Poids du CNN sauvegardes dans : {chemin_modele}")
    
    resultats = {
        'historique': historique,
        'y_vrai': y_vrai,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'temps_entrainement': temps_entrainement,
        'nb_parametres': nb_parametres
    }
    
    return resultats, modele
