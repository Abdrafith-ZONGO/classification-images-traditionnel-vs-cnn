import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from src.config import REPORTS_DIR
import pandas as pd

def tracer_courbes_apprentissage(historique, nom_dataset, sauver=True):
    """
    on trace et on sauvegarde les courbes de perte (Loss) et de precision (Accuracy)
    de l'entrainement et de la validation du CNN.
    """
    epochs = range(1, len(historique['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Graphe de la perte
    plt.subplot(1, 2, 1)
    plt.plot(epochs, historique['train_loss'], label='Entrainement', color='#F472B6', linewidth=2)
    plt.plot(epochs, historique['test_loss'], label='Test/Val', color='#6C9FFF', linestyle='--', linewidth=2)
    plt.title('Courbe de Perte (Loss)')
    plt.xlabel('Epoques')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Graphe de la precision
    plt.subplot(1, 2, 2)
    plt.plot(epochs, historique['train_acc'], label='Entrainement', color='#34D399', linewidth=2)
    plt.plot(epochs, historique['test_acc'], label='Test/Val', color='#6C9FFF', linestyle='--', linewidth=2)
    plt.title('Courbe de Precision (Accuracy)')
    plt.xlabel('Epoques')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.suptitle(f"Suivi d'apprentissage du CNN sur {nom_dataset}", fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if sauver:
        chemin_sauvegarde = os.path.join(REPORTS_DIR, f"courbe_apprentissage_{nom_dataset}.png")
        plt.savefig(chemin_sauvegarde, bbox_inches='tight', dpi=150)
        print(f"   - Courbes d'apprentissage sauvegardees dans : {chemin_sauvegarde}")
        
    plt.show()
    plt.close()


def tracer_matrice_confusion(y_vrai, y_pred, classes, nom_modele, nom_dataset, sauver=True):
    """
    on affiche et on sauvegarde la matrice de confusion sous forme de carte thermique.
    """
    cm = confusion_matrix(y_vrai, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predit')
    plt.ylabel('Reel')
    plt.title(f"Matrice de Confusion\n{nom_modele} sur {nom_dataset}")
    plt.tight_layout()
    
    if sauver:
        nom_sans_espace = nom_modele.replace(" ", "_").replace("(", "").replace(")", "")
        chemin_sauvegarde = os.path.join(REPORTS_DIR, f"confusion_{nom_sans_espace}_{nom_dataset}.png")
        plt.savefig(chemin_sauvegarde, bbox_inches='tight', dpi=150)
        
    plt.show()
    plt.close()


def tracer_courbe_roc(y_vrai, y_prob, classes, nom_modele, nom_dataset, sauver=True):
    """
    on genere la courbe ROC et on calcule le score AUC.
    on s'adapte au cas binaire et au cas multi-classes avec l'approche One-vs-Rest.
    """
    nb_classes = len(classes)
    
    if y_prob is None:
        return
        
    plt.figure(figsize=(6, 5))
    
    if nb_classes == 2:
        # Cas binaire (ex: Covid19 ou Wildfire)
        if len(y_prob.shape) == 2:
            prob_positive = y_prob[:, 1]
        else:
            prob_positive = y_prob
            
        fpr, tpr, _ = roc_curve(y_vrai, prob_positive)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='#FF5733', lw=2, label=f"AUC = {roc_auc:.4f}")
        
    else:
        # Cas multi-classes (ex: Iris ou DTD) avec approche One-vs-Rest
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

    # Diagonale du hasard
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title(f"Courbe ROC-AUC : {nom_modele}\nDataset : {nom_dataset}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if sauver:
        nom_sans_espace = nom_modele.replace(" ", "_").replace("(", "").replace(")", "")
        chemin_sauvegarde = os.path.join(REPORTS_DIR, f"roc_{nom_sans_espace}_{nom_dataset}.png")
        plt.savefig(chemin_sauvegarde, bbox_inches='tight', dpi=150)
        
    plt.show()
    plt.close()


def tracer_comparaison_globale(df_comparatif, sauver=True):
    """
    on genere des graphiques a barres horizontaux comparant les precisions des modeles.
    les resultats s'affichent sous forme de pourcentages et le CNN est mis en couleur pour ressortir.
    """
    if df_comparatif.empty:
        return
        
    # on copie les donnees et on les convertit en pourcentages
    df = df_comparatif.copy()
    df['Accuracy_pct'] = df['Accuracy'] * 100
    
    # on trie par dataset et par precision decroissante
    df_tri = df.sort_values(by=['Dataset', 'Accuracy_pct'], ascending=[True, False])
    
    # style minimaliste
    sns.set_theme(style='white', context='talk')
    
    # Bleu vif pour le CNN personnalise, gris pour les autres
    palette = {mod: "#3A86FF" if mod == "CNN Personnalisé" else "#8D99AE" for mod in df_tri['Modele'].unique()}
    
    g = sns.catplot(
        data=df_tri, kind='bar',
        x='Accuracy_pct', y='Modele', col='Dataset',
        col_wrap=2, height=3.8, aspect=1.5,
        palette=palette, hue='Modele', dodge=False, legend=False
    )
    
    g.set_axis_labels('Precision Globale (%)', '')
    g.set_titles('Jeu de donnees : {col_name}', fontweight='bold', size=12)
    
    for ax in g.axes.flat:
        ax.set_xlim(0, 105)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        sns.despine(ax=ax, left=True, bottom=True)
        # on affiche les valeurs de precision en pourcentage
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=4, size=10)
            
    plt.suptitle('Comparaison des Performances (ML Traditionnel vs CNN)', y=1.02, fontweight='bold', fontsize=16)
    plt.tight_layout()
    
    if sauver:
        chemin_sauvegarde = os.path.join(REPORTS_DIR, "comparatif_global_accuracy.png")
        plt.savefig(chemin_sauvegarde, bbox_inches='tight', dpi=300)
        print(f"   - Graphique comparatif par dataset sauvegarde dans : {chemin_sauvegarde}")
        
    plt.show()
    plt.close()
    
    # === CLASSEMENT ULTIME MOYEN ===
    df_mean = df.groupby('Modele')['Accuracy_pct'].mean().reset_index()
    df_mean = df_mean.sort_values(by='Accuracy_pct', ascending=False)
    
    plt.figure(figsize=(10, 5.5))
    sns.set_theme(style='white', context='talk')
    
    palette_finale = {mod: "#3A86FF" if mod == "CNN Personnalisé" else "#8D99AE" for mod in df_mean['Modele']}
    
    ax = sns.barplot(data=df_mean, x='Accuracy_pct', y='Modele', palette=palette_finale, hue='Modele', dodge=False, legend=False)
    
    plt.xlim(0, 105)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    sns.despine(left=True, bottom=True)
    
    plt.title("CLASSEMENT FINAL GLOBAL\nPrecision moyenne sur l'ensemble des jeux de donnees", fontweight='bold', fontsize=16, pad=15)
    plt.xlabel("Precision Moyenne (%)", fontweight='bold', labelpad=10)
    plt.ylabel("")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=5, size=11, fontweight='bold')
        
    plt.tight_layout()
    
    if sauver:
        chemin_classement = os.path.join(REPORTS_DIR, "classement_final_moyen.png")
        plt.savefig(chemin_classement, bbox_inches='tight', dpi=300)
        print(f"   - Graphique du classement final sauvegarde dans : {chemin_classement}")
        
    sns.reset_orig()
    plt.show()
    plt.close()
