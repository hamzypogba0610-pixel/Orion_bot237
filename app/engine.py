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

def calc_shadow_market(proba_ensemble, cote_bookmaker):
    """
    Compare notre probabilité avec celle du bookmaker.
    Retourne l'écart (Edge) et le pourcentage de marge implicite.
    """
    if cote_bookmaker <= 1.0:
        return {"erreur": "Cote invalide"}
        
    proba_implicite = (1 / cote_bookmaker) * 100
    
    # On estime la marge du bookmaker (Vig) à environ 5% en moyenne
    marge_bookmaker = 5.0
    proba_reelle_bookmaker = proba_implicite - (proba_implicite * (marge_bookmaker / 100))
    
    # Edge (Notre avantage)
    edge = proba_ensemble - proba_reelle_bookmaker
    
    if edge > 5.0:
        statut = "🔥 VALUE BET (Le marché est aveugle)"
    elif edge > 0:
        statut = "✅ Léger avantage"
    else:
        statut = "❌ Pas d'avantage (Le marché a raison)"

    return {
        "cote_saisie": cote_bookmaker,
        "proba_implicite_brute": round(proba_implicite, 2),
        "proba_reelle_bookmaker_estimee": round(proba_reelle_bookmaker, 2),
        "notre_proba_ensemble": round(proba_ensemble, 2),
        "edge": round(edge, 2),
        "statut_shadow_market": statut
    }

def calc_live_corners(moyenne_dom, moyenne_ext, minute_actuelle, corners_actuels, score_dom, score_ext, ligne):
    """
    Calcule la probabilité de corners en Live en tenant compte du temps restant et du score.
    """
    # 1. Temps restant (on considère 90 minutes de jeu)
    temps_restant = max(0, 90 - minute_actuelle)
    
    # 2. Intensité de base (Corners par minute)
    # Moyenne totale attendue sur le match / 90
    intensite_par_minute = (moyenne_dom + moyenne_ext) / 90
    
    # 3. Ajustement selon le score (L'équipe qui perd pousse plus)
    facteur_enjeu = 1.0
    ecart_buts = abs(score_dom - score_ext)
    
    if ecart_buts == 0:
        facteur_enjeu = 1.05 # Match nul, on pousse un peu
    elif ecart_buts == 1:
        facteur_enjeu = 1.15 # 1 but d'écart, l'équipe qui perd attaque fort (+15%)
    else:
        facteur_enjeu = 1.25 # 2 buts d'écart, jeu plus ouvert ou équipe qui abandonne (+25%)
        
    # 4. Calcul du Lambda pour le temps restant
    lambda_restant = temps_restant * intensite_par_minute * facteur_enjeu
    
    # 5. Combien de corners manque-t-il pour dépasser la ligne ?
    # Si ligne = 9.5 et qu'il y a 5 corners, il en manque 5 pour atteindre 10.
    corners_manquants = max(0, ligne - corners_actuels)
    
    # 6. Probabilité avec Poisson
    # On veut P(X > corners_manquants), donc 1 - P(X <= corners_manquants)
    p_live = (1 - poisson.cdf(int(np.floor(corners_manquants)), lambda_restant)) * 100
    
    return {
        "temps_restant_min": temps_restant,
        "intensite_par_minute": round(intensite_par_minute, 3),
        "facteur_enjeu": facteur_enjeu,
        "lambda_restant": round(lambda_restant, 2),
        "corners_manquants": corners_manquants,
        "probabilite_live": round(p_live, 2)
    }

def calc_kalman_filter(historique_corners):
    """
    Filtre de Kalman 1D pour lisser la vraie moyenne d'une équipe.
    Ignore les matchs aberrants (bruit) pour trouver la tendance de fond.
    """
    if not historique_corners:
        return 0.0
    if len(historique_corners) < 3:
        return np.mean(historique_corners)

    # Paramètres du filtre
    x = historique_corners[0]  # Estimation initiale (premier match)
    P = 1.0                    # Incertitude initiale
    Q = 0.1                    # Bruit du processus (à quel point la vraie moyenne change)
    R = 5.0                    # Bruit de la mesure (à quel point les matchs sont variables)

    for z in historique_corners[1:]:
        # 1. Prédiction
        x_pred = x
        P_pred = P + Q

        # 2. Mise à jour (Correction avec la nouvelle observation)
        K = P_pred / (P_pred + R)  # Gain de Kalman
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred

    return x

def calc_markov_impact(historique_corners):
    """
    Calcule la probabilité de transition vers un état 'Tempête' (beaucoup de corners).
    Utilise une version simplifiée de la Chaîne de Markov Cachée.
    """
    if len(historique_corners) < 4:
        return 1.0 # Pas assez de données pour détecter une transition
    
    moyenne = np.mean(historique_corners)
    
    # Définition des états : 0 (Calme), 1 (Normal), 2 (Tempête)
    etats = []
    for c in historique_corners:
        if c < moyenne * 0.7:
            etats.append(0)
        elif c > moyenne * 1.3:
            etats.append(2)
        else:
            etats.append(1)
            
    # On regarde l'état actuel (le dernier match)
    etat_actuel = etats[-1]
    
    # On calcule la probabilité de transition de l'état actuel vers l'état 2 (Tempête)
    transitions_possibles = 0
    transitions_vers_tempete = 0
    
    for i in range(len(etats) - 1):
        if etats[i] == etat_actuel:
            transitions_possibles += 1
            if etats[i+1] == 2:
                transitions_vers_tempete += 1
                
    if transitions_possibles == 0:
        return 1.0
        
    proba_tempete = transitions_vers_tempete / transitions_possibles
    
    # On crée un multiplicateur basé sur cette probabilité
    # Si proba_tempete > 0.5, on augmente les corners attendus
    multiplicateur = 1.0 + (proba_tempete * 0.20) # Max +20% si 100% de chances de tempête
    
    return min(multiplicateur, 1.20)

def calc_bivariate_poisson_matrix(lambda_dom, lambda_ext, max_buts=5):
    """
    Calcule la matrice des probabilités de tous les scores exacts (0-0 à 5-5).
    Utilise une correction de Dixon-Coles pour les scores faibles.
    """
    # Facteur de corrélation (rho). Une valeur négative augmente la proba des 0-0, 1-0, 0-1, 1-1
    rho = -0.05 
    
    matrix = []
    
    for i in range(max_buts + 1):
        for j in range(max_buts + 1):
            # Probabilité de base (Poisson indépendant)
            p_base = poisson.pmf(i, lambda_dom) * poisson.pmf(j, lambda_ext)
            
            # Correction de Dixon-Coles pour les petits scores
            facteur = 1.0
            if i == 0 and j == 0:
                facteur = 1 - (lambda_dom * lambda_ext * rho)
            elif i == 1 and j == 0:
                facteur = 1 + (lambda_ext * rho)
            elif i == 0 and j == 1:
                facteur = 1 + (lambda_dom * rho)
            elif i == 1 and j == 1:
                facteur = 1 - rho
                
            proba_finale = p_base * facteur
            matrix.append({"score": f"{i}-{j}", "proba": proba_finale})
            
    # On trie pour avoir les scores les plus probables en premier
    matrix.sort(key=lambda x: x["proba"], reverse=True)
    return matrix

def calc_time_intensity(minute, lambda_total_match):
    """
    Calcule un multiplicateur d'intensité en fonction de la minute.
    Les buts sont plus probables en fin de match (fatigue, prise de risque).
    """
    if minute <= 15:
        facteur = 0.85 # Début de match, on s'observe
    elif minute <= 30:
        facteur = 0.95
    elif minute <= 45:
        facteur = 1.05 # Juste avant la mi-temps
    elif minute <= 60:
        facteur = 1.10
    elif minute <= 75:
        facteur = 1.20 # Le match s'ouvre
    else:
        facteur = 1.35 # Money time, tout peut arriver
        
    return facteur

def apply_copula_tail_dependence(score_dom, score_ext, proba_base, lambda_total):
    """
    Applique un ajustement de copule (dépendance de queue).
    Augmente légèrement la probabilité des scores extrêmes (ex: 3-0, 4-2).
    """
    multiplicateur = 1.0
    
    # Si le match a un potentiel de buts élevé (lambda_total > 3)
    if lambda_total > 3.0:
        ecart = abs(score_dom - score_ext)
        total_buts = score_dom + score_ext
        
        # Si c'est un score large (ex: écart de 3 buts ou total >= 4 buts)
        if ecart >= 3 or total_buts >= 4:
            multiplicateur = 1.15 # On booste la proba de ces scores de 15%
            
    return proba_base * multiplicateur

# ==========================================
# 5. LES COUCHES 4 ET 5 (Viterbi & EVT)
# ==========================================

def calc_viterbi_state(minute, score_dom, score_ext):
    """
    Couche 4 : Modèle d'état (Viterbi simplifié).
    Détermine si le match est dans un état Prudent, Offensif ou Chaos.
    Retourne un multiplicateur d'intensité.
    """
    ecart = abs(score_dom - score_ext)
    total_buts = score_dom + score_ext
    
    # Règle de transition d'état
    if minute < 30:
        return 0.9 # Prudent, on s'observe
    elif minute < 70:
        if ecart == 0 and total_buts == 0:
            return 1.1 # On commence à pousser
        elif ecart <= 1:
            return 1.2 # Match ouvert
        else:
            return 1.0 # Match plié, on gère
    else: # Fin de match (70+)
        if ecart == 0:
            return 1.4 # Tout le monde attaque pour gagner
        elif ecart == 1:
            return 1.3 # L'équipe qui perd pousse
        else:
            return 1.1 # Match fini, mais quelques contres possibles

def apply_evt(score_dom, score_ext, proba_base, lambda_total):
    """
    Couche 5 : Théorie des Valeurs Extrêmes (EVT).
    Applique une distribution de Pareto pour rehausser les scores très larges.
    """
    total_buts = score_dom + score_ext
    ecart = abs(score_dom - score_ext)
    
    multiplicateur = 1.0
    
    # Si le total de buts est extrême (5 buts ou plus)
    if total_buts >= 5:
        multiplicateur += 0.20 # +20%
    # Si l'écart est extrême (4 buts ou plus)
    elif ecart >= 4:
        multiplicateur += 0.15 # +15%
        
    return proba_base * multiplicateur

# ==========================================
# 6. LA COUCHE 6 (Market Drift & Live Arbitrage)
# ==========================================

def calc_live_score_drift(proba_model, cote_bookmaker, marge_bookmaker=0.15):
    """
    Détecte les anomalies de cotes en direct pour les scores exacts.
    La marge d'un bookmaker sur un score exact est en général de 15% à 20%.
    """
    if cote_bookmaker <= 1.0:
        return {"erreur": "Cote invalide"}

    # Probabilité implicite brute de la cote
    proba_implicite = (1 / cote_bookmaker) * 100
    
    # On retire la marge du bookmaker pour trouver sa "vraie" probabilité
    proba_reelle_bookmaker = proba_implicite * (1 - marge_bookmaker)
    
    # Calcul de l'Edge (Notre avantage)
    edge = proba_model - proba_reelle_bookmaker

    if edge > 5.0:
        statut = "🔥 VALUE BET LIVE (Drift détecté)"
    elif edge > 0:
        statut = "✅ Léger avantage"
    else:
        statut = "❌ Pas d'avantage (Le marché a raison)"

    return {
        "cote_saisie": cote_bookmaker,
        "proba_implicite_brute": round(proba_implicite, 2),
        "proba_reelle_bookmaker_estimee": round(proba_reelle_bookmaker, 2),
        "notre_proba_model": round(proba_model, 2),
        "edge": round(edge, 2),
        "statut_drift": statut
    }
