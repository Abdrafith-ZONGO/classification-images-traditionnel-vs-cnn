import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from src.config import REPORTS_DIR

def tracer_courbes_apprentissage(historique, nom_dataset, sauver=True):
    """
    Trace et sauvegarde les courbes de perte (Loss) et d'exactitude (Accuracy)
    pour l'entraînement et le test du CNN personnalisé.
    """
    epochs = range(1, len(historique['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Graphe 1 : Perte (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, historique['train_loss'], label='Entraînement', color='#F472B6', linewidth=2)
    plt.plot(epochs, historique['test_loss'], label='Test/Val', color='#6C9FFF', linestyle='--', linewidth=2)
    plt.title('Courbe de Perte (Loss)')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Graphe 2 : Exactitude (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, historique['train_acc'], label='Entraînement', color='#34D399', linewidth=2)
    plt.plot(epochs, historique['test_acc'], label='Test/Val', color='#6C9FFF', linestyle='--', linewidth=2)
    plt.title("Courbe d'Exactitude (Accuracy)")
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.suptitle(f"Suivi d'apprentissage du CNN — {nom_dataset}", fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if sauver:
        chemin_sauvegarde = os.path.join(REPORTS_DIR, f"courbe_apprentissage_{nom_dataset}.png")
        plt.savefig(chemin_sauvegarde, bbox_inches='tight', dpi=150)
        print(f"   -> Graphique des courbes d'apprentissage sauvegardé sous : {chemin_sauvegarde}")
        
    plt.show()
    plt.close()


def tracer_matrice_confusion(y_vrai, y_pred, classes, nom_modele, nom_dataset, sauver=True):
    """
    Génère et sauvegarde la matrice de confusion sous forme de carte de chaleur (heatmap).
    """
    cm = confusion_matrix(y_vrai, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title(f"Matrice de Confusion\n{nom_modele} sur {nom_dataset}")
    plt.tight_layout()
    
    if sauver:
        nom_sans_espace = nom_modele.replace(" ", "_")
        chemin_sauvegarde = os.path.join(REPORTS_DIR, f"confusion_{nom_sans_espace}_{nom_dataset}.png")
        plt.savefig(chemin_sauvegarde, bbox_inches='tight', dpi=150)
        
    plt.show()
    plt.close()


def tracer_courbe_roc(y_vrai, y_prob, classes, nom_modele, nom_dataset, sauver=True):
    """
    Génère et sauvegarde la courbe ROC (Receiver Operating Characteristic) et calcule l'AUC.
    Gère automatiquement le cas binaire et le cas multi-classe (via approche One-vs-Rest).
    """
    nb_classes = len(classes)
    
    # Si nous n'avons pas de probabilités (ex. certains classifieurs sans predict_proba), on passe
    if y_prob is None:
        return
        
    plt.figure(figsize=(6, 5))
    
    if nb_classes == 2:
        # Cas binaire (ex. Covid19-XRAYS ou Wildfire)
        # y_prob peut être de dimension (n_samples, 2), on prend la probabilité de la classe positive (index 1)
        if len(y_prob.shape) == 2:
            prob_positive = y_prob[:, 1]
        else:
            prob_positive = y_prob
            
        fpr, tpr, _ = roc_curve(y_vrai, prob_positive)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='#FF5733', lw=2, label=f"AUC = {roc_auc:.4f}")
        
    else:
        # Cas multi-classe (ex. DTD ou Iris) - Approche One-vs-Rest
        # Binarisation des labels pour le calcul
        y_vrai_bin = label_binarize(y_vrai, classes=list(range(nb_classes)))
        
        for i in range(nb_classes):
            fpr, tpr, _ = roc_curve(y_vrai_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, label=f"Classe {classes[i]} (AUC = {roc_auc:.3f})")
            
        # Calcul de la moyenne macro des ROC
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(nb_classes):
            fpr, tpr, _ = roc_curve(y_vrai_bin[:, i], y_prob[:, i])
            mean_tpr += np.interp(fpr_grid, fpr, tpr)
        mean_tpr /= nb_classes
        macro_auc = auc(fpr_grid, mean_tpr)
        
        plt.plot(fpr_grid, mean_tpr, color='black', linestyle=':', lw=2, label=f"Macro-Moyenne (AUC = {macro_auc:.3f})")

    # Diagonale de référence (modèle aléatoire)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title(f"Courbe ROC-AUC — {nom_modele}\nDataset : {nom_dataset}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if sauver:
        nom_sans_espace = nom_modele.replace(" ", "_")
        chemin_sauvegarde = os.path.join(REPORTS_DIR, f"roc_{nom_sans_espace}_{nom_dataset}.png")
        plt.savefig(chemin_sauvegarde, bbox_inches='tight', dpi=150)
        
    plt.show()
    plt.close()
