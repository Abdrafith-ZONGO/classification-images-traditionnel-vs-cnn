import time
import os
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config import SEED, MODELS_DIR

def obtenir_grille_parametres(nom_classifieur):
    """
    Définit les grilles d'hyperparamètres pour l'optimisation par validation croisée (GridSearchCV).
    """
    if nom_classifieur == "SVM":
        return {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 0.001, 0.01]  # Pertinent uniquement pour le noyau rbf
        }
    elif nom_classifieur == "k-NN":
        return {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    elif nom_classifieur == "Arbre de Decision":
        return {
            'max_depth': [3, 5, 10, None],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 5]
        }
    elif nom_classifieur == "Naïve Bayes":
        # Le Naïve Bayes a peu d'hyperparamètres à optimiser, on fait varier la variance de lissage
        return {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    else:
        raise ValueError(f"Classifieur inconnu : {nom_classifieur}")


def instancier_classifieur_base(nom_classifieur):
    """
    Instancie le classifieur avec les valeurs par défaut et la graine aléatoire.
    """
    if nom_classifieur == "SVM":
        return SVC(probability=True, random_state=SEED) # probability=True requis pour la courbe ROC
    elif nom_classifieur == "k-NN":
        return KNeighborsClassifier()
    elif nom_classifieur == "Arbre de Decision":
        return DecisionTreeClassifier(random_state=SEED)
    elif nom_classifieur == "Naïve Bayes":
        return GaussianNB()
    else:
        raise ValueError(f"Classifieur inconnu : {nom_classifieur}")


def entrainer_evaluer_classique(nom_dataset, nom_modele_extraction, nom_classifieur, X_train, y_train, X_test, y_test, optimiser=True):
    """
    Entraîne un classifieur classique sur les features extraites,
    effectue l'optimisation des hyperparamètres par GridSearchCV,
    calcule les métriques de performance et sauvegarde le modèle entraîné.
    
    Justification méthodologique : La normalisation standard (StandardScaler) est appliquée sur
    le train set et propagée sur le test set pour éviter toute fuite de données (data leakage).
    """
    print(f"Entraînement de {nom_classifieur} sur les features de {nom_modele_extraction} ({nom_dataset})...")
    
    # 1. Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Choix du classifieur de base
    clf_base = instancier_classifieur_base(nom_classifieur)
    
    debut_temps = time.time()
    
    # 3. Entraînement et GridSearchCV
    if optimiser:
        grille = obtenir_grille_parametres(nom_classifieur)
        # Validation croisée 3-fold pour ne pas surcharger les calculs
        grid_search = GridSearchCV(
            estimator=clf_base,
            param_grid=grille,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train_scaled, y_train)
        meilleur_modele = grid_search.best_estimator_
        meilleurs_params = grid_search.best_params_
        print(f"   -> Meilleurs paramètres trouvés : {meilleurs_params}")
    else:
        clf_base.fit(X_train_scaled, y_train)
        meilleur_modele = clf_base
        meilleurs_params = "defaut"
        
    temps_entrainement = time.time() - debut_temps
    print(f"   -> Temps d'entraînement : {temps_entrainement:.2f} secondes")
    
    # 4. Prédictions
    y_pred = meilleur_modele.predict(X_test_scaled)
    
    # Calcul des probabilités pour ROC-AUC (si le modèle le permet)
    y_prob = None
    if hasattr(meilleur_modele, "predict_proba"):
        y_prob = meilleur_modele.predict_proba(X_test_scaled)
        
    # 5. Calcul des métriques de performance
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # 6. Sauvegarde du modèle et du scaler associé
    nom_fichier_modele = f"model_{nom_dataset}_{nom_modele_extraction}_{nom_classifieur}.pkl"
    chemin_sauvegarde = os.path.join(MODELS_DIR, nom_fichier_modele)
    
    dict_sauvegarde = {
        'modele': meilleur_modele,
        'scaler': scaler,
        'hyperparametres': meilleurs_params,
        'metriques': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'temps_entrainement': temps_entrainement
        }
    }
    joblib.dump(dict_sauvegarde, chemin_sauvegarde)
    
    print(f"   -> Modèle sauvegardé dans {chemin_sauvegarde}")
    print(f"   -> Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    
    return dict_sauvegarde, y_pred, y_prob
