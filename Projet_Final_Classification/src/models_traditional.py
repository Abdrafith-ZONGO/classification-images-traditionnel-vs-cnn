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
    on prepare les grilles pour la recherche des meilleurs hyperparametres avec GridSearchCV.
    """
    if nom_classifieur == "SVM":
        return {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 0.001, 0.01]  # pour le noyau rbf uniquement
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
        # le classifieur Gaussien de Bayes a peu de reglages, on ajuste le lissage
        return {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    else:
        raise ValueError(f"Classifieur inconnu : {nom_classifieur}")


def instancier_classifieur_base(nom_classifieur):
    """
    on cree le classifieur avec ses options par defaut.
    """
    if nom_classifieur == "SVM":
        # probability=True est indispensable pour pouvoir tracer la courbe ROC ensuite
        return SVC(probability=True, random_state=SEED)
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
    on normalise les caracteristiques puis on entraine le modele classique.
    si optimiser=True, on cherche les meilleurs reglages avec une validation croisee 3-fold.
    on calcule ensuite les metriques et on enregistre le modele et son scaler lie.
    """
    print(f"Entrainement de {nom_classifieur} sur les features de {nom_modele_extraction} ({nom_dataset})...")
    
    # on applique une normalisation standard (centrer et reduire)
    # on calcule les moyennes sur l'entrainement puis on applique sur train et test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf_base = instancier_classifieur_base(nom_classifieur)
    
    debut_temps = time.time()
    
    # entrainement du modele
    if optimiser:
        grille = obtenir_grille_parametres(nom_classifieur)
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
        print(f"   - Meilleurs parametres trouves : {meilleurs_params}")
    else:
        clf_base.fit(X_train_scaled, y_train)
        meilleur_modele = clf_base
        meilleurs_params = "defaut"
        
    temps_entrainement = time.time() - debut_temps
    print(f"   - Temps d'entrainement : {temps_entrainement:.2f} secondes")
    
    # predictions
    y_pred = meilleur_modele.predict(X_test_scaled)
    
    # on recupere les probabilites pour pouvoir tracer la courbe ROC
    y_prob = None
    if hasattr(meilleur_modele, "predict_proba"):
        y_prob = meilleur_modele.predict_proba(X_test_scaled)
        
    # calcul des metriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # on range le modele, le scaler et les resultats dans un fichier .pkl
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
    
    print(f"   - Modele et scaler sauvegardes dans : {chemin_sauvegarde}")
    print(f"   - Resultats : Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")
    
    return dict_sauvegarde, y_pred, y_prob
