import os
import matplotlib
matplotlib.use('Agg')
import argparse
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
from src.config import DATASET_PATHS, DATASET_CLASSES, TAILLE_IMAGE_CNN
from src.dataset import preparer_loaders
from src.extractors import obtenir_features_dataset
from src.models_traditional import entrainer_evaluer_classique
from src.models_cnn import entrainer_evaluer_cnn, CNNPerso
from src.utils import tracer_courbes_apprentissage, tracer_matrice_confusion, tracer_courbe_roc

def executer_pipeline_complet(datasets_a_traiter, modeles_extracteurs, classifieurs_ml, epochs_cnn, forcer_extraction=False):
    """
    Exécute l'intégralité du pipeline d'apprentissage et de comparaison
    pour les datasets et modèles sélectionnés.
    """
    resultats_comparatifs = []
    
    for nom_dataset in datasets_a_traiter:
        print("\n" + "="*80)
        print(f"=== DEBUT DU TRAITEMENT DU JEU DE DONNEES : {nom_dataset} ===")
        print("="*80)
        
        # --- PHASE 1 : Pipeline Traditionnel (Extraction + ML) ---
        print("\n--- PHASE 1 : Extraction de Features & Classifieurs Classiques ---")
        
        for nom_extractor in modeles_extracteurs:
            # 1. Extraction ou chargement des caractéristiques
            try:
                X_train, y_train, X_test, y_test = obtenir_features_dataset(
                    nom_dataset, nom_extractor, forcer_reextraction=forcer_extraction
                )
            except Exception as e:
                print(f"[ATTENTION] Impossible d'extraire les caractéristiques pour {nom_dataset} avec {nom_extractor} : {e}")
                continue
                
            # 2. Entraînement de chaque classifieur ML
            for nom_clf in classifieurs_ml:
                try:
                    res_clf, y_pred, y_prob = entrainer_evaluer_classique(
                        nom_dataset, nom_extractor, nom_clf,
                        X_train, y_train, X_test, y_test,
                        optimiser=True
                    )
                    
                    # Récupération du nombre de paramètres du modèle d'extraction de features
                    # (pour le tableau comparatif final)
                    if nom_extractor == "AlexNet":
                        modele_ext = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
                    elif nom_extractor == "VGG16":
                        modele_ext = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
                    elif nom_extractor == "InceptionV3":
                        modele_ext = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
                    
                    nb_params_ext = sum(p.numel() for p in modele_ext.parameters())
                    
                    # Tracé des visualisations
                    classes = DATASET_CLASSES[nom_dataset]
                    nom_complet_modele = f"{nom_clf} (via {nom_extractor})"
                    
                    tracer_matrice_confusion(y_test, y_pred, classes, nom_complet_modele, nom_dataset)
                    tracer_courbe_roc(y_test, y_prob, classes, nom_complet_modele, nom_dataset)
                    
                    # Stockage des résultats
                    resultats_comparatifs.append({
                        'Dataset': nom_dataset,
                        'Pipeline': 'Pipeline 1 (Traditionnel)',
                        'Modele': nom_complet_modele,
                        'Accuracy': res_clf['metriques']['accuracy'],
                        'Precision': res_clf['metriques']['precision'],
                        'Recall': res_clf['metriques']['recall'],
                        'F1-Score': res_clf['metriques']['f1'],
                        'Temps_Entrainement_Sec': res_clf['metriques']['temps_entrainement'],
                        'Nb_Parametres': nb_params_ext # Nombre de paramètres de l'extracteur
                    })
                    
                except Exception as e:
                    print(f"[ATTENTION] Echec de l'entrainement de {nom_clf} sur {nom_dataset} avec {nom_extractor} : {e}")
                    
        # --- PHASE 2 : Pipeline CNN (Entraînement direct) ---
        print("\n--- PHASE 2 : Entraînement direct du CNN personnalisé ---")
        try:
            # 1. Chargement des loaders pour le CNN (taille d'image spécifique)
            loader_train, loader_test, classes = preparer_loaders(nom_dataset, TAILLE_IMAGE_CNN)
            
            # 2. Entraînement du CNN personnalisé
            res_cnn, modele_trained = entrainer_evaluer_cnn(
                nom_dataset, loader_train, loader_test, len(classes), epochs=epochs_cnn
            )
            
            # Calcul des métriques de test finales
            y_vrai = res_cnn['y_vrai']
            y_pred = res_cnn['y_pred']
            y_prob = res_cnn['y_prob']
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(y_vrai, y_pred)
            prec = precision_score(y_vrai, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_vrai, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_vrai, y_pred, average='weighted', zero_division=0)
            
            # Tracé des courbes d'apprentissage, ROC et Matrice de confusion
            tracer_courbes_apprentissage(res_cnn['historique'], nom_dataset)
            tracer_matrice_confusion(y_vrai, y_pred, classes, "CNN Personnalisé", nom_dataset)
            tracer_courbe_roc(y_vrai, y_prob, classes, "CNN Personnalisé", nom_dataset)
            
            # Stockage des résultats du CNN
            resultats_comparatifs.append({
                'Dataset': nom_dataset,
                'Pipeline': 'Pipeline 2 (CNN Perso)',
                'Modele': 'CNN Personnalisé',
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'Temps_Entrainement_Sec': res_cnn['temps_entrainement'],
                'Nb_Parametres': res_cnn['nb_parametres']
            })
            
        except Exception as e:
            print(f"[ATTENTION] Echec du Pipeline CNN personnalise sur {nom_dataset} : {e}")

    # --- PHASE 3 : Comparaison et enregistrement des résultats ---
    if resultats_comparatifs:
        df_comparatif = pd.DataFrame(resultats_comparatifs)
        
        # Sauvegarde du tableau final au format CSV
        chemin_csv_global = os.path.abspath(os.path.join(os.path.dirname(__file__), "reports/comparaison_globale.csv"))
        df_comparatif.to_csv(chemin_csv_global, index=False)
        
        print("\n" + "="*80)
        print("=== TABLEAU COMPARATIF FINAL GENERE ET ENREGISTRE ===")
        print("="*80)
        print(df_comparatif.to_string(index=False))
        print(f"\nResultats sauvegardes dans : {chemin_csv_global}")
    else:
        print("Aucun resultat n'a pu etre collecte.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Classification d'Images - ML classique vs CNN")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["all", "Covid19-XRAYS", "DTD", "Iris", "Wildfire"], 
        default="Iris", 
        help="Dataset à traiter (choisir 'all' pour les exécuter tous, par défaut : 'Iris')"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=15, 
        help="Nombre d'époques d'entraînement pour le CNN personnalisé (par défaut : 15)"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Forcer l'extraction des features même si les fichiers .npy existent déjà"
    )
    
    args = parser.parse_args()
    
    # Choix des datasets
    if args.dataset == "all":
        datasets_a_traiter = ["Iris", "Covid19-XRAYS", "Wildfire", "DTD"]
    else:
        datasets_a_traiter = [args.dataset]
        
    # Liste des modèles et classifieurs requis par l'énoncé
    modeles_extracteurs = ["AlexNet", "VGG16", "InceptionV3"]
    classifieurs_ml = ["SVM", "k-NN", "Arbre de Decision", "Naïve Bayes"]
    
    executer_pipeline_complet(
        datasets_a_traiter, 
        modeles_extracteurs, 
        classifieurs_ml, 
        args.epochs, 
        forcer_extraction=args.force
    )
