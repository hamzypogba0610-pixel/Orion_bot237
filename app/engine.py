import numpy as np
from scipy.stats import poisson, nbinom
from datetime import datetime

# ==========================================
# 1. MÉTHODES STATISTIQUES CLASSIQUES
# ==========================================

def calc_poisson(lambda_total, ligne):
    """Calcule la probabilité P(X > ligne) selon la loi de Poisson."""
    # On arrondit à l'entier inférieur pour le cdf (ex: 9.5 -> 9)
    return (1 - poisson.cdf(int(np.floor(ligne)), lambda_total)) * 100

def calc_binomiale_negative(lambda_total, ligne, alpha=2.5):
    """Calcule la probabilité avec la Binomiale Négative (gère mieux les extrêmes)."""
    n = 1 / alpha
    p = 1 / (1 + alpha * lambda_total)
    return (1 - nbinom.cdf(int(np.floor(ligne)), n, p)) * 100

def calc_monte_carlo(lambda_total, ligne, simulations=10000):
    """Simule le match 10 000 fois et compte les réussites."""
    sim = np.random.poisson(lambda_total, simulations)
    return np.mean(sim > ligne) * 100

# ==========================================
# 2. CALCUL DES LAMBDAS AVANCÉS
# ==========================================

def calc_lambda_dixon_coles(matchs_equipe):
    """
    Calcule la moyenne pondérée par le temps (Dixon-Coles).
    matchs_equipe est une liste de dictionnaires: [{'corners': 5, 'date': datetime}, ...]
    """
    poids_total = 0
    corners_ponderes = 0
    maintenant = datetime.utcnow()
    
    for match in matchs_equipe:
        # Décroissance exponentielle : plus c'est vieux, moins ça compte
        # (0.005 = un match d'il y a 100 jours ne vaut plus que 60% d'un match récent)
        jours_ecoules = (maintenant - match['date']).days if match['date'] else 30
        poids = np.exp(-0.005 * jours_ecoules)
        
        corners_ponderes += match['corners'] * poids
        poids_total += poids
        
    return (corners_ponderes / poids_total) if poids_total > 0 else 0

def calc_lambda_bayesien(corners_equipe, nb_matchs, moyenne_championnat, force_rappel=5):
    """
    Calcule la moyenne ajustée par l'approche Bayésienne (Shrinkage).
    Si l'équipe a peu de matchs (ex: 2), on la "tire" vers la moyenne du championnat.
    """
    if nb_matchs == 0:
        return moyenne_championnat / 2
        
    # Plus force_rappel est grand, plus on fait confiance au championnat sur les petits échantillons
    moyenne_bayes = (corners_equipe + (force_rappel * (moyenne_championnat / 2))) / (nb_matchs + force_rappel)
    return moyenne_bayes

# ==========================================
# 3. LES MODÈLES "ALIENS"
# ==========================================

def calc_hawkes_impact(historique_corners):
    """
    Calcule le 'facteur d'auto-excitation' d'une équipe.
    Si une équipe a une variance très élevée par rapport à sa moyenne,
    cela signifie qu'elle fait des séries (des vagues).
    """
    if len(historique_corners) < 3:
        return 1.0 # Pas assez de données, on ne change rien
    
    moyenne = np.mean(historique_corners)
    variance = np.var(historique_corners)
    
    if moyenne == 0: return 1.0
    
    # Coefficient de dispersion
    dispersion = variance / moyenne 
    
    # Si la dispersion est forte (ex: matchs à 2 corners et matchs à 12 corners),
    # l'effet Hawkes est important.
    # On crée un multiplicateur (ex: si dispersion = 2, multiplicateur = 1.1)
    multiplicateur = 1.0 + (dispersion * 0.05)
    
    # On plafonne pour éviter des prédictions absurdes
    return min(multiplicateur, 1.25) 
