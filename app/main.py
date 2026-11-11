from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import httpx
import numpy as np
from scipy.stats import poisson
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, Competition, Match, MatchStat, Prediction
from app.engine import (
    calc_poisson, calc_binomiale_negative, calc_monte_carlo,
    calc_lambda_dixon_coles, calc_lambda_bayesien, calc_hawkes_impact,
    calc_shadow_market, calc_live_corners, calc_kalman_filter,
    calc_markov_impact, calc_time_intensity, apply_copula_tail_dependence,
    calc_bivariate_poisson_matrix, calc_viterbi_state, apply_evt,
    calc_live_score_drift, calc_zero_inflated_poisson, calc_mmpp,
    calc_live_par, calc_hamilton_regime, calc_hurst_exponent, 
    calc_live_cards, calc_merton_jump_diffusion,
    calibrer_probabilite, calc_intervalle_confiance
)
from datetime import datetime, timedelta
import asyncio
import csv
import io
import hashlib

app = FastAPI(title="ORION", version="0.1.0")
templates = Jinja2Templates(directory="templates")

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/matchs")
async def api_matchs():
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        matchs = db.query(Match).order_by(Match.date_heure.desc()).limit(50).all()
        resultat = []
        for m in matchs:
            resultat.append({
    "id": m.id,
    "nom": f"{m.equipe_domicile} vs {m.equipe_exterieur}",
    "date": m.date_heure.strftime("%d/%m/%Y") if m.date_heure else "Date inconnue",
    "logo_dom": m.logo_domicile,
    "logo_ext": m.logo_exterieur
})
        return resultat
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/api/historique")
async def api_historique():
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        predictions = db.query(Prediction, Match).join(Match, Prediction.match_id == Match.id).order_by(Prediction.date_prediction.desc()).limit(20).all()
        resultat = []
        for p, m in predictions:
            resultat.append({
                "id": p.id,
                "match": f"{m.equipe_domicile} vs {m.equipe_exterieur}",
                "marche": p.marche,
                "ligne": p.ligne,
                "probabilite": p.probabilite_brute,
                "date": p.date_prediction.strftime("%d/%m/%Y %H:%M") if p.date_prediction else "Date inconnue",
                "statut": p.statut_prediction,
                "resultat_reel": p.resultat_reel
            })
        return resultat
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/update-schema")
async def update_schema():
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        db.execute(text("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS arbitre VARCHAR;"))
        db.execute(text("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS arbitre_profil VARCHAR;"))
        db.commit()
        return {"message": "✅ Les colonnes 'arbitre' et 'arbitre_profil' ont été ajoutées !"}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()

@app.post("/sauvegarder-prediction/{match_id_local}")
async def sauvegarder_prediction(
    match_id_local: int, 
    p_corners: float = 0.0, 
    p_cartons: float = 0.0,
    meteo: str = "Sec",
    style_adv: str = "Équilibré",
    arbitre: str = "Non renseigné",
    arbitre_profil: str = "Normal"
):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        # ON AJOUTE order_by(Prediction.id.desc()).first() ICI
        pred_c = db.query(Prediction).filter(Prediction.match_id == match_id_local, Prediction.marche == "corners").order_by(Prediction.id.desc()).first()
        pred_k = db.query(Prediction).filter(Prediction.match_id == match_id_local, Prediction.marche == "cartons").order_by(Prediction.id.desc()).first()

        if pred_c:
            pred_c.probabilite_brute = p_corners
            pred_c.meteo = meteo
            pred_c.style_adversaire = style_adv
            pred_c.arbitre = arbitre
            pred_c.arbitre_profil = arbitre_profil
            pred_c.date_prediction = datetime.utcnow()
        else:
            db.add(Prediction(
                match_id=match_id_local, marche="corners", ligne=9.5,
                probabilite_brute=p_corners, probabilite_calibree=p_corners,
                modele_version="poisson-v1.0", date_prediction=datetime.utcnow(),
                meteo=meteo, style_adversaire=style_adv, arbitre=arbitre, arbitre_profil=arbitre_profil
            ))

        if pred_k:
            pred_k.probabilite_brute = p_cartons
            pred_k.meteo = meteo
            pred_k.style_adversaire = style_adv
            pred_k.arbitre = arbitre
            pred_k.arbitre_profil = arbitre_profil
            pred_k.date_prediction = datetime.utcnow()
        else:
            db.add(Prediction(
                match_id=match_id_local, marche="cartons", ligne=4.5,
                probabilite_brute=p_cartons, probabilite_calibree=p_cartons,
                modele_version="poisson-v1.0", date_prediction=datetime.utcnow(),
                meteo=meteo, style_adversaire=style_adv, arbitre=arbitre, arbitre_profil=arbitre_profil
            ))

        db.commit()
        return {"message": "Analyse enregistrée (ou mise à jour) avec succès !"}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/init-competitions")
async def init_competitions():
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        les_ligues = [
    {"id_api": 39, "nom": "Premier League", "pays": "Angleterre"},
    {"id_api": 140, "nom": "La Liga", "pays": "Espagne"},
    {"id_api": 135, "nom": "Serie A", "pays": "Italie"},
    {"id_api": 78, "nom": "Bundesliga", "pays": "Allemagne"},
    {"id_api": 61, "nom": "Ligue 1", "pays": "France"},
    {"id_api": 88, "nom": "Eredivisie", "pays": "Pays-Bas"},
    {"id_api": 94, "nom": "Primeira Liga", "pays": "Portugal"},
    {"id_api": 144, "nom": "Pro League", "pays": "Belgique"},
    {"id_api": 203, "nom": "Süper Lig", "pays": "Turquie"},
    {"id_api": 197, "nom": "Super League", "pays": "Grèce"},
    {"id_api": 2, "nom": "Ligue des Champions", "pays": "Europe"},
    {"id_api": 3, "nom": "Europa League", "pays": "Europe"},
    {"id_api": 71, "nom": "Brasileirao", "pays": "Brésil"},
    {"id_api": 307, "nom": "Saudi Pro League", "pays": "Arabie Saoudite"},
    {"id_api": 253, "nom": "MLS", "pays": "USA"}
        ]
        ajoutees = 0
        for ligue in les_ligues:
            existe = db.query(Competition).filter(Competition.id_source == ligue["id_api"]).first()
            if not existe:
                db.add(Competition(nom=ligue["nom"], pays=ligue["pays"], id_source=ligue["id_api"]))
                ajoutees += 1
        db.commit()
        return {"message": f"{ajoutees} compétitions ajoutées."}
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/predict-corners/{match_id_local}")
async def predict_corners(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # 1. Calcul de la moyenne GLOBALE du championnat (pour les corners)
        tous_les_matchs = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.competition_id == match.competition_id).all()
        
        total_corners_championnat, nb_matchs_championnat = 0, 0
        for stat, m in tous_les_matchs:
            if stat.corners_domicile is not None and stat.corners_exterieur is not None:
                total_corners_championnat += (stat.corners_domicile + stat.corners_exterieur)
                nb_matchs_championnat += 1
        
        moyenne_championnat = (total_corners_championnat / nb_matchs_championnat) if nb_matchs_championnat > 0 else 10.0

        # 2. Force d'attaque à domicile (Combien l'équipe A marque de corners par rapport à la moyenne)
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_domicile)
        ).all()
        corners_dom, nb_matchs_dom = 0, 0
        for stat, m in stats_dom:
            if stat.corners_domicile is not None:
                corners_dom += stat.corners_domicile
                nb_matchs_dom += 1
        moyenne_dom = (corners_dom / nb_matchs_dom) if nb_matchs_dom > 0 else moyenne_championnat
        force_attaque_dom = moyenne_dom / (moyenne_championnat / 2) # divisé par 2 car moyenne_championnat est le total des 2 équipes

        # 3. Force de défense à l'extérieur (Combien l'équipe B concède de corners par rapport à la moyenne)
        stats_def_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_exterieur == match.equipe_exterieur)
        ).all()
        corners_concedes_ext, nb_matchs_def_ext = 0, 0
        for stat, m in stats_def_ext:
            if stat.corners_domicile is not None:
                corners_concedes_ext += stat.corners_domicile
                nb_matchs_def_ext += 1
        moyenne_concedee_ext = (corners_concedes_ext / nb_matchs_def_ext) if nb_matchs_def_ext > 0 else moyenne_championnat / 2
        force_defense_ext = moyenne_concedee_ext / (moyenne_championnat / 2)

        # 4. Force d'attaque à l'extérieur (Equipe B)
        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_exterieur == match.equipe_exterieur)
        ).all()
        corners_ext, nb_matchs_ext = 0, 0
        for stat, m in stats_ext:
            if stat.corners_exterieur is not None:
                corners_ext += stat.corners_exterieur
                nb_matchs_ext += 1
        moyenne_ext = (corners_ext / nb_matchs_ext) if nb_matchs_ext > 0 else moyenne_championnat / 2
        force_attaque_ext = moyenne_ext / (moyenne_championnat / 2)

        # 5. Force de défense à domicile (Equipe A)
        stats_def_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_domicile)
        ).all()
        corners_concedes_dom, nb_matchs_def_dom = 0, 0
        for stat, m in stats_def_dom:
            if stat.corners_exterieur is not None:
                corners_concedes_dom += stat.corners_exterieur
                nb_matchs_def_dom += 1
        moyenne_concedee_dom = (corners_concedes_dom / nb_matchs_def_dom) if nb_matchs_def_dom > 0 else moyenne_championnat / 2
        force_defense_dom = moyenne_concedee_dom / (moyenne_championnat / 2)

        # 6. Calcul du Lambda final (Attendu A = Moyenne_Champ/2 * Force_Attaque_A * Force_Defense_B)
        lambda_dom = (moyenne_championnat / 2) * force_attaque_dom * force_defense_ext
        lambda_ext = (moyenne_championnat / 2) * force_attaque_ext * force_defense_dom
        lambda_total = lambda_dom + lambda_ext

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "contexte": {
                "moyenne_championnat": round(moyenne_championnat, 2),
                "force_attaque_dom": round(force_attaque_dom, 2),
                "force_defense_ext": round(force_defense_ext, 2)
            },
            "analyse_corners": {
                "attendu_domicile": round(lambda_dom, 2), 
                "attendu_exterieur": round(lambda_ext, 2),
                "total_attendu": round(lambda_total, 2),
                "probabilite_plus_de_9_5": round((1 - poisson.cdf(9, lambda_total)) * 100, 2),
                "probabilite_plus_de_8_5": round((1 - poisson.cdf(8, lambda_total)) * 100, 2)
            }
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/predict-cards/{match_id_local}")
async def predict_cards(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # 1. Moyenne globale du championnat (Cartons)
        tous_les_matchs = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.competition_id == match.competition_id).all()
        total_cartons_championnat, nb_matchs_championnat = 0, 0
        for stat, m in tous_les_matchs:
            if stat.cartons_jaunes_domicile is not None and stat.cartons_jaunes_exterieur is not None:
                total_cartons_championnat += (stat.cartons_jaunes_domicile + stat.cartons_jaunes_exterieur)
                nb_matchs_championnat += 1
        moyenne_championnat = (total_cartons_championnat / nb_matchs_championnat) if nb_matchs_championnat > 0 else 4.5

        # 2. Force d'attaque à domicile (Cartons pris par A)
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_domicile == match.equipe_domicile).all()
        cartons_dom, nb_matchs_dom = 0, 0
        for stat, m in stats_dom:
            if stat.cartons_jaunes_domicile is not None:
                cartons_dom += stat.cartons_jaunes_domicile
                nb_matchs_dom += 1
        moyenne_dom = (cartons_dom / nb_matchs_dom) if nb_matchs_dom > 0 else moyenne_championnat / 2
        force_attaque_dom = moyenne_dom / (moyenne_championnat / 2)

        # 3. Force de défense à l'extérieur (Cartons subis par l'équipe B à l'extérieur)
        stats_def_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_exterieur == match.equipe_exterieur).all()
        cartons_subis_ext, nb_matchs_def_ext = 0, 0
        for stat, m in stats_def_ext:
            if stat.cartons_jaunes_domicile is not None: 
                cartons_subis_ext += stat.cartons_jaunes_domicile
                nb_matchs_def_ext += 1
        moyenne_subis_ext = (cartons_subis_ext / nb_matchs_def_ext) if nb_matchs_def_ext > 0 else moyenne_championnat / 2
        force_defense_ext = moyenne_subis_ext / (moyenne_championnat / 2)

        # 4. Force d'attaque à l'extérieur (Cartons pris par B)
        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_exterieur == match.equipe_exterieur).all()
        cartons_ext, nb_matchs_ext = 0, 0
        for stat, m in stats_ext:
            if stat.cartons_jaunes_exterieur is not None:
                cartons_ext += stat.cartons_jaunes_exterieur
                nb_matchs_ext += 1
        moyenne_ext = (cartons_ext / nb_matchs_ext) if nb_matchs_ext > 0 else moyenne_championnat / 2
        force_attaque_ext = moyenne_ext / (moyenne_championnat / 2)

        # 5. Force de défense à domicile (Cartons subis par l'équipe A à domicile)
        stats_def_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_domicile == match.equipe_domicile).all()
        cartons_subis_dom, nb_matchs_def_dom = 0, 0
        for stat, m in stats_def_dom:
            if stat.cartons_jaunes_exterieur is not None:
                cartons_subis_dom += stat.cartons_jaunes_exterieur
                nb_matchs_def_dom += 1
        moyenne_subis_dom = (cartons_subis_dom / nb_matchs_def_dom) if nb_matchs_def_dom > 0 else moyenne_championnat / 2
        force_defense_dom = moyenne_subis_dom / (moyenne_championnat / 2)

        # 6. Calcul final
        lambda_dom = (moyenne_championnat / 2) * force_attaque_dom * force_defense_ext
        lambda_ext = (moyenne_championnat / 2) * force_attaque_ext * force_defense_dom
        lambda_total = lambda_dom + lambda_ext

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "contexte": {
                "moyenne_championnat": round(moyenne_championnat, 2),
                "force_attaque_dom": round(force_attaque_dom, 2),
                "force_defense_ext": round(force_defense_ext, 2)
            },
            "analyse_cartons": {
                "attendu_domicile": round(lambda_dom, 2),
                "attendu_exterieur": round(lambda_ext, 2),
                "total_attendu": round(lambda_total, 2),
                "probabilite_plus_de_3_5": round((1 - poisson.cdf(3, lambda_total)) * 100, 2),
                "probabilite_plus_de_4_5": round((1 - poisson.cdf(4, lambda_total)) * 100, 2),
                "probabilite_plus_de_5_5": round((1 - poisson.cdf(5, lambda_total)) * 100, 2)
            }
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/predict-ht-score/{match_id_local}")
async def predict_ht_score(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # 1. Moyennes de buts
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter((Match.equipe_domicile == match.equipe_domicile) | (Match.equipe_exterieur == match.equipe_domicile)).all()
        buts_dom, nb_dom = 0, 0
        for stat, m in stats_dom:
            if m.equipe_domicile == match.equipe_domicile and m.score_domicile is not None: buts_dom += m.score_domicile; nb_dom += 1
            elif m.equipe_exterieur == match.equipe_domicile and m.score_exterieur is not None: buts_dom += m.score_exterieur; nb_dom += 1
        moy_buts_dom = (buts_dom / nb_dom) if nb_dom > 0 else 1.5

        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter((Match.equipe_domicile == match.equipe_exterieur) | (Match.equipe_exterieur == match.equipe_exterieur)).all()
        buts_ext, nb_ext = 0, 0
        for stat, m in stats_ext:
            if m.equipe_domicile == match.equipe_exterieur and m.score_domicile is not None: buts_ext += m.score_domicile; nb_ext += 1
            elif m.equipe_exterieur == match.equipe_exterieur and m.score_exterieur is not None: buts_ext += m.score_exterieur; nb_ext += 1
        moy_buts_ext = (buts_ext / nb_ext) if nb_ext > 0 else 1.2

        # 2. Couches Magiques
        # Intensité temporelle
        facteur_temps = calc_time_intensity(minute=45, lambda_total_match=(moy_buts_dom + moy_buts_ext))
        
        # Viterbi (État du match à la mi-temps, score initial 0-0)
        etat_viterbi = calc_viterbi_state(minute=45, score_dom=0, score_ext=0)
        
        lambda_dom_ajuste = (moy_buts_dom / 2) * facteur_temps * etat_viterbi
        lambda_ext_ajuste = (moy_buts_ext / 2) * facteur_temps * etat_viterbi

        matrice = calc_bivariate_poisson_matrix(lambda_dom_ajuste, lambda_ext_ajuste, max_buts=3)
        
        scores_ajustes = []
        for s in matrice:
            i, j = map(int, s["score"].split("-"))
            # Application de la copule
            proba_copule = apply_copula_tail_dependence(i, j, s["proba"], lambda_dom_ajuste + lambda_ext_ajuste)
            # Application de l'EVT
            proba_finale = apply_evt(i, j, proba_copule, lambda_dom_ajuste + lambda_ext_ajuste)
            scores_ajustes.append({"score": s["score"], "proba": proba_finale})
            
        scores_ajustes.sort(key=lambda x: x["proba"], reverse=True)
        top_3 = scores_ajustes[:3]

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "top_scores_mi_temps": [{"score": s["score"], "probabilite": round(s["proba"] * 100, 2)} for s in top_3]
        }
    except Exception as e: return {"erreur": str(e)}
    finally: db.close()

@app.get("/import-csv/")
async def import_csv(url: str):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        nom_ligue = ""
        if "E0.csv" in url: nom_ligue = "Premier League"
        elif "SP1.csv" in url: nom_ligue = "La Liga"
        elif "I1.csv" in url: nom_ligue = "Serie A"
        elif "F1.csv" in url: nom_ligue = "Ligue 1"
        elif "D1.csv" in url: nom_ligue = "Bundesliga"
        elif "N1.csv" in url: nom_ligue = "Eredivisie"
        elif "P1.csv" in url: nom_ligue = "Primeira Liga"
        elif "B1.csv" in url: nom_ligue = "Pro League"
        elif "T1.csv" in url: nom_ligue = "Süper Lig"
        elif "G1.csv" in url: nom_ligue = "Super League"
        elif "BRA.csv" in url: nom_ligue = "Brasileirao"
        elif "USA.csv" in url: nom_ligue = "MLS"
        else: return {"erreur": "Lien non reconnu."}

        competition_locale = db.query(Competition).filter(Competition.nom == nom_ligue).first()
        if not competition_locale:
            return {"erreur": f"La compétition {nom_ligue} n'est pas dans la base."}
        
        competition_id = competition_locale.id 

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
        
        if response.status_code != 200: 
            return {"erreur": f"Erreur HTTP : {response.status_code}. Vérifie l'URL."}
            
        csv_file = io.StringIO(response.text)
        reader = csv.DictReader(csv_file)
        
        matchs_ajoutes = 0
        stats_ajoutees = 0

        for row in reader:
            # Gestion des deux formats de colonnes (Standard et Extra)
            home_team = row.get('HomeTeam') or row.get('Home')
            away_team = row.get('AwayTeam') or row.get('Away')
            date_val = row.get('Date')
            
            if not home_team or not away_team or not date_val: 
                continue
                
            fake_id_str = f"{home_team}{away_team}{date_val}"
            fake_id = int(hashlib.md5(fake_id_str.encode()).hexdigest()[:8], 16) % 2000000000
            
            if db.query(Match).filter(Match.id_source_apifootball == fake_id).first(): 
                continue

            try:
                date_obj = datetime.strptime(date_val, "%d/%m/%Y")
            except:
                try:
                    date_obj = datetime.strptime(date_val, "%d/%m/%y")
                except:
                    date_obj = datetime.now()

            # Gestion des scores pour les deux formats
            s_dom = row.get('FTHG') or row.get('HG')
            s_ext = row.get('FTAG') or row.get('AG')

            nouveau_match = Match(
                competition_id=competition_id, date_heure=date_obj,
                equipe_domicile=home_team, equipe_exterieur=away_team, statut="FT",
                id_source_apifootball=fake_id,
                score_domicile=int(s_dom) if s_dom and s_dom.isdigit() else None,
                score_exterieur=int(s_ext) if s_ext and s_ext.isdigit() else None
            )
            db.add(nouveau_match)
            db.commit()
            db.refresh(nouveau_match)
            matchs_ajoutes += 1

            # Les stats (HC, AC, etc.) ont les mêmes noms dans les deux formats
            if row.get('HC') and row.get('AC'):
                nouvelle_stat = MatchStat(
                    match_id=nouveau_match.id, source_api="csv-football-data",
                    corners_domicile=int(row['HC']) if row['HC'].isdigit() else 0,
                    corners_exterieur=int(row['AC']) if row['AC'].isdigit() else 0,
                    cartons_jaunes_domicile=int(row['HY']) if row.get('HY') and row['HY'].isdigit() else 0,
                    cartons_jaunes_exterieur=int(row['AY']) if row.get('AY') and row['AY'].isdigit() else 0,
                    cartons_rouges_domicile=int(row['HR']) if row.get('HR') and row['HR'].isdigit() else 0,
                    cartons_rouges_exterieur=int(row['AR']) if row.get('AR') and row['AR'].isdigit() else 0,
                    statut_validation="confirmé"
                )
                db.add(nouvelle_stat)
                stats_ajoutees += 1

        db.commit()
        return {"message": f"Importation réussie ! {matchs_ajoutes} matchs et {stats_ajoutees} stats ajoutés pour la {nom_ligue}."}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/sync-today")
async def sync_today():
    api_key = os.getenv("FIVE_DOLLAR_KEY")
    if not api_key or not SessionLocal: return {"erreur": "Configuration manquante"}
    
    db = SessionLocal()
    try:
        aujourd_hui = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        demain = aujourd_hui + timedelta(days=1)
        start_ts = int(aujourd_hui.timestamp())
        end_ts = int(demain.timestamp())
        
        url = f"https://api.5dollarfootballapi.com/v1/fixtures?start_time={start_ts}&end_time={end_ts}"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            
        if response.status_code != 200:
            return {"erreur": f"Erreur API : {response.status_code}"}
            
        data = response.json()
        matchs_api = data.get("data", [])
        
        if not matchs_api: return {"message": "Aucun match trouvé aujourd'hui."}

        total_matchs_ajoutes = 0

        for m in matchs_api:
            nom_ligue_api = m["league"]["name"].lower()
            
            competition_locale = None
            for comp in db.query(Competition).all():
                if comp.nom.lower() in nom_ligue_api:
                    competition_locale = comp
                    break
            
            if competition_locale:
                id_match_api = m["id"]
                
                match_local = db.query(Match).filter(Match.id_source_apifootball == id_match_api).first()
                if not match_local:
                    date_str = m["kickoff_utc"].replace("Z", "+00:00")
                    
                    nouveau_match = Match(
                        competition_id=competition_locale.id,
                        date_heure=datetime.fromisoformat(date_str),
                        equipe_domicile=m["teams"]["home"]["name"],
                        equipe_exterieur=m["teams"]["away"]["name"],
                        statut="NS",
                        id_source_apifootball=id_match_api,
                        score_domicile=m.get("goals", {}).get("home"),
                        score_exterieur=m.get("goals", {}).get("away")
                    )
                    db.add(nouveau_match)
                    total_matchs_ajoutes += 1
        
        db.commit()
        return {"message": f"Succès ! {total_matchs_ajoutes} matchs ajoutés depuis la nouvelle API."}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close() 

@app.get("/predict-ft-score/{match_id_local}")
async def predict_ft_score(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # 1. Moyennes de buts
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter((Match.equipe_domicile == match.equipe_domicile) | (Match.equipe_exterieur == match.equipe_domicile)).all()
        buts_dom, nb_dom = 0, 0
        for stat, m in stats_dom:
            if m.equipe_domicile == match.equipe_domicile and m.score_domicile is not None: buts_dom += m.score_domicile; nb_dom += 1
            elif m.equipe_exterieur == match.equipe_domicile and m.score_exterieur is not None: buts_dom += m.score_exterieur; nb_dom += 1
        moy_buts_dom = (buts_dom / nb_dom) if nb_dom > 0 else 1.5

        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter((Match.equipe_domicile == match.equipe_exterieur) | (Match.equipe_exterieur == match.equipe_exterieur)).all()
        buts_ext, nb_ext = 0, 0
        for stat, m in stats_ext:
            if m.equipe_domicile == match.equipe_exterieur and m.score_domicile is not None: buts_ext += m.score_domicile; nb_ext += 1
            elif m.equipe_exterieur == match.equipe_exterieur and m.score_exterieur is not None: buts_ext += m.score_exterieur; nb_ext += 1
        moy_buts_ext = (buts_ext / nb_ext) if nb_ext > 0 else 1.2

        # 2. Application des Couches Magiques
        # Intensité temporelle (90 min)
        facteur_temps = calc_time_intensity(minute=90, lambda_total_match=(moy_buts_dom + moy_buts_ext))
        
        # Viterbi (État du match à la 90ème, score initial 0-0)
        etat_viterbi = calc_viterbi_state(minute=90, score_dom=0, score_ext=0)
        
        lambda_dom_ajuste = moy_buts_dom * facteur_temps * etat_viterbi
        lambda_ext_ajuste = moy_buts_ext * facteur_temps * etat_viterbi

        # Matrice Bivariate Poisson (Dixon-Coles)
        matrice = calc_bivariate_poisson_matrix(lambda_dom_ajuste, lambda_ext_ajuste, max_buts=5)
        
        scores_ajustes = []
        for s in matrice:
            i, j = map(int, s["score"].split("-"))
            # Application de la copule
            proba_copule = apply_copula_tail_dependence(i, j, s["proba"], lambda_dom_ajuste + lambda_ext_ajuste)
            # Application de l'EVT (Scores extrêmes)
            proba_finale = apply_evt(i, j, proba_copule, lambda_dom_ajuste + lambda_ext_ajuste)
            scores_ajustes.append({"score": s["score"], "proba": proba_finale})
            
        scores_ajustes.sort(key=lambda x: x["proba"], reverse=True)
        top_3 = scores_ajustes[:3]

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "top_scores_finaux": [{"score": s["score"], "probabilite": round(s["proba"] * 100, 2)} for s in top_3]
        }
    except Exception as e: return {"erreur": str(e)}
    finally: db.close()

@app.get("/clean-duplicates")
async def clean_duplicates():
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        # On utilise une commande SQL directe pour supprimer les doublons
        # On garde uniquement l'ID le plus récent (le plus grand) pour chaque match et chaque marché
        sql = text("""
            DELETE FROM predictions a USING predictions b
            WHERE a.id < b.id 
            AND a.match_id = b.match_id 
            AND a.marche = b.marche;
        """)
        result = db.execute(sql)
        db.commit()
        return {"message": f"Ménage terminé ! {result.rowcount} anciens doublons ont été supprimés."}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/api/calibration")
async def api_calibration():
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        # On récupère uniquement les prédictions qui ont un résultat
        toutes_predictions = db.query(Prediction).filter(Prediction.statut_prediction != 'en_attente').all()
        
        total = len(toutes_predictions)
        if total == 0: return {"message": "Pas encore assez de résultats pour calculer une calibration."}
        
        gagnees = sum(1 for p in toutes_predictions if p.statut_prediction == 'gagne')
        taux_reussite_reel = (gagnees / total) * 100
        
        # On calcule la probabilité moyenne annoncée par le bot
        proba_moyenne_annoncee = sum(p.probabilite_brute for p in toutes_predictions) / total
        
        # On regarde l'écart
        ecart = taux_reussite_reel - proba_moyenne_annoncee
        
        statut = "Bien calibré ✅"
        if ecart < -5: statut = "Trop confiant ⚠️ (baissez les probas)"
        elif ecart > 5: statut = "Trop prudent 📈 (augmentez les probas)"

        return {
            "total_predictions_analysees": total,
            "gagnees": gagnees,
            "perdues": total - gagnees,
            "taux_reussite_reel": round(taux_reussite_reel, 2),
            "probabilite_moyenne_annoncee": round(proba_moyenne_annoncee, 2),
            "ecart": round(ecart, 2),
            "diagnostic_calibration": statut
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/predict-ensemble/{match_id_local}")
async def predict_ensemble(match_id_local: int, meteo: str = "Sec", cote: float = 1.85):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # 1. Moyenne du championnat
        tous_les_matchs = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.competition_id == match.competition_id).all()
        total_corners_champ, nb_matchs_champ = 0, 0
        for stat, m in tous_les_matchs:
            if stat.corners_domicile is not None and stat.corners_exterieur is not None:
                total_corners_champ += (stat.corners_domicile + stat.corners_exterieur)
                nb_matchs_champ += 1
        moy_championnat = (total_corners_champ / nb_matchs_champ) if nb_matchs_champ > 0 else 10.0

        # 2. Historiques
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_domicile == match.equipe_domicile).all()
        hist_dom = []
        total_corners_dom, nb_dom = 0, 0
        for stat, m in stats_dom:
            if stat.corners_domicile is not None:
                hist_dom.append({"corners": stat.corners_domicile, "date": m.date_heure})
                total_corners_dom += stat.corners_domicile
                nb_dom += 1
        
        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_exterieur == match.equipe_exterieur).all()
        hist_ext = []
        total_corners_ext, nb_ext = 0, 0
        for stat, m in stats_ext:
            if stat.corners_exterieur is not None:
                hist_ext.append({"corners": stat.corners_exterieur, "date": m.date_heure})
                total_corners_ext += stat.corners_exterieur
                nb_ext += 1

        # 3. Lambdas classiques
        lambda_dc_total = (calc_lambda_dixon_coles(hist_dom) if nb_dom > 0 else moy_championnat / 2) + (calc_lambda_dixon_coles(hist_ext) if nb_ext > 0 else moy_championnat / 2)
        lambda_bayes_total = calc_lambda_bayesien(total_corners_dom, nb_dom, moy_championnat) + calc_lambda_bayesien(total_corners_ext, nb_ext, moy_championnat)
        lambda_simple_total = ((total_corners_dom / nb_dom) if nb_dom > 0 else moy_championnat / 2) + ((total_corners_ext / nb_ext) if nb_ext > 0 else moy_championnat / 2)

        impact_meteo = 0.90 if meteo == "Pluie" else (0.85 if meteo == "Vent fort" else 1.0)
        lambda_simple_total *= impact_meteo
        lambda_dc_total *= impact_meteo
        lambda_bayes_total *= impact_meteo

        # 4. Hawkes & Kalman & Markov
        liste_corners_dom = [h['corners'] for h in hist_dom]
        liste_corners_ext = [h['corners'] for h in hist_ext]
        
        impact_hawkes_total = (calc_hawkes_impact(liste_corners_dom) + calc_hawkes_impact(liste_corners_ext)) / 2
        lambda_hawkes_total = lambda_simple_total * impact_hawkes_total

        lambda_kalman_total = (calc_kalman_filter(liste_corners_dom) if nb_dom > 0 else moy_championnat / 2) + (calc_kalman_filter(liste_corners_ext) if nb_ext > 0 else moy_championnat / 2)
        lambda_kalman_total *= impact_meteo

        impact_markov_dom = calc_markov_impact(liste_corners_dom)
        impact_markov_ext = calc_markov_impact(liste_corners_ext)
        lambda_markov_total = (((total_corners_dom / nb_dom) if nb_dom > 0 else moy_championnat / 2) * impact_markov_dom + ((total_corners_ext / nb_ext) if nb_ext > 0 else moy_championnat / 2) * impact_markov_ext) * impact_meteo

        # 5. LES INNOVATIONS
        ligne = 9.5
        p_zip = calc_zero_inflated_poisson(lambda_simple_total, ligne, zero_prob=0.05)
        etat = calc_viterbi_state(minute=45, score_dom=0, score_ext=0)
        lambda_mmpp = calc_mmpp(lambda_simple_total, etat)
        p_mmpp = calc_poisson(lambda_mmpp, ligne)
        recent_corners = liste_corners_dom[-3:] + liste_corners_ext[-3:] 
        p_par, _ = calc_live_par(recent_corners, temps_restant=10, ligne=2.5)
        mult_ham_dom, _ = calc_hamilton_regime(liste_corners_dom)
        mult_ham_ext, _ = calc_hamilton_regime(liste_corners_ext)
        lambda_hamilton = (((total_corners_dom / nb_dom) if nb_dom > 0 else moy_championnat / 2) * mult_ham_dom + ((total_corners_ext / nb_ext) if nb_ext > 0 else moy_championnat / 2) * mult_ham_ext) * impact_meteo
        p_hamilton = calc_poisson(lambda_hamilton, ligne)
        hurst_val, _ = calc_hurst_exponent(liste_corners_dom + liste_corners_ext)
        p_hurst = calc_poisson(lambda_simple_total * (1 + (hurst_val - 0.5)), ligne)

        # 6. Calcul des probabilités des modèles de base
        p_poisson = calc_poisson(lambda_simple_total, ligne)
        p_nbinom = calc_binomiale_negative(lambda_simple_total, ligne)
        p_mc = calc_monte_carlo(lambda_simple_total, ligne)
        p_dc = calc_poisson(lambda_dc_total, ligne)
        p_bayes = calc_poisson(lambda_bayes_total, ligne)
        p_hawkes = calc_poisson(lambda_hawkes_total, ligne)
        p_kalman = calc_poisson(lambda_kalman_total, ligne)
        p_markov = calc_poisson(lambda_markov_total, ligne)

        # --- 7. NOUVEAU : MERTON JUMP DIFFUSION ---
        p_merton = calc_merton_jump_diffusion(lambda_simple_total, ligne)

        # 8. ENSEMBLE FINAL (15 MODÈLES)
        liste_probas = [
            p_poisson, p_nbinom, p_mc, p_dc, p_bayes, p_hawkes, p_kalman, p_markov,
            p_zip, p_mmpp, p_par, p_hamilton, p_hurst, p_merton
        ]
        
        poids = [1]*8 + [1.5]*6
        moyenne_ensemble = np.average(liste_probas, weights=poids)
        ecart_type = np.std(liste_probas)

        modeles_proches = sum(1 for p in liste_probas if abs(p - moyenne_ensemble) <= 10)
        
        if modeles_proches >= 11: fiabilite = "HAUTE ✅"
        elif modeles_proches >= 8: fiabilite = "MOYENNE ⚠️"
        else: fiabilite = "FAIBLE ❌ (Divergence)"

        proba_calibree = calibrer_probabilite(moyenne_ensemble, ecart_type, modeles_proches)
        shadow_result = calc_shadow_market(moyenne_ensemble, cote)

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "moyennes_attendues": {
                "simple": round(lambda_simple_total, 2),
                "dixon_coles": round(lambda_dc_total, 2),
                "bayesien": round(lambda_bayes_total, 2),
                "hawkes": round(lambda_hawkes_total, 2),
                "kalman": round(lambda_kalman_total, 2),
                "markov": round(lambda_markov_total, 2)
            },
            "probabilites_ligne_9_5": {
                "poisson": round(p_poisson, 2),
                "binomiale_negative": round(p_nbinom, 2),
                "monte_carlo": round(p_mc, 2),
                "dixon_coles": round(p_dc, 2),
                "bayesien": round(p_bayes, 2),
                "hawkes": round(p_hawkes, 2),
                "kalman": round(p_kalman, 2),
                "markov": round(p_markov, 2),
                "zip": round(p_zip, 2),
                "mmpp": round(p_mmpp, 2),
                "par": round(p_par, 2),
                "hamilton": round(p_hamilton, 2),
                "hurst": round(p_hurst, 2),
                "merton": round(p_merton, 2) # <--- NOUVEAU
            },
            "ensemble_final": round(moyenne_ensemble, 2),
            "proba_calibree": round(proba_calibree, 2),
            "ecart_type": round(ecart_type, 2),
            "fiabilite": fiabilite,
            "shadow_market": shadow_result
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/predict-live/{match_id_local}")
async def predict_live(
    match_id_local: int, 
    minute: int, 
    corners_dom: int, 
    corners_ext: int, 
    score_dom: int, 
    score_ext: int, 
    ligne: float, 
    cote: float
):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # 1. Récupération des moyennes historiques
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_domicile == match.equipe_domicile).all()
        corners_d, nb_d = 0, 0
        for stat, m in stats_dom:
            if stat.corners_domicile is not None:
                corners_d += stat.corners_domicile; nb_d += 1
        moy_dom = (corners_d / nb_d) if nb_d > 0 else 5.5

        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_exterieur == match.equipe_exterieur).all()
        corners_e, nb_e = 0, 0
        for stat, m in stats_ext:
            if stat.corners_exterieur is not None:
                corners_e += stat.corners_exterieur; nb_e += 1
        moy_ext = (corners_e / nb_e) if nb_e > 0 else 4.0

        # 2. Calcul Live
        corners_actuels_total = corners_dom + corners_ext
        resultat_live = calc_live_corners(
            moyenne_dom=moy_dom, 
            moyenne_ext=moy_ext, 
            minute_actuelle=minute, 
            corners_actuels=corners_actuels_total, 
            score_dom=score_dom, 
            score_ext=score_ext, 
            ligne=ligne
        )

        # 3. Calcul de la Value (EV) avec la cote saisie
        proba_decimale = resultat_live["probabilite_live"] / 100
        ev = (proba_decimale * cote) - 1
        
        if ev > 0:
            verdict = f"🔥 VALUE BET LIVE (+{round(ev*100, 1)}%)"
        else:
            verdict = f"❌ Pas de Value ({round(ev*100, 1)}%)"

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "etat_actuel": {
                "minute": minute,
                "score": f"{score_dom} - {score_ext}",
                "corners_actuels": corners_actuels_total,
                "ligne_visee": ligne,
                "cote_saisie": cote
            },
            "analyse_live": resultat_live,
            "verdict_value": verdict
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/predict-live-score/{match_id_local}")
async def predict_live_score(
    match_id_local: int, 
    minute: int, 
    score_dom: int, 
    score_ext: int, 
    score_pari: str, # Ex: "2-1"
    cote_score: float
):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # 1. Moyennes de buts (Base)
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter((Match.equipe_domicile == match.equipe_domicile) | (Match.equipe_exterieur == match.equipe_domicile)).all()
        buts_dom, nb_dom = 0, 0
        for stat, m in stats_dom:
            if m.equipe_domicile == match.equipe_domicile and m.score_domicile is not None: buts_dom += m.score_domicile; nb_dom += 1
            elif m.equipe_exterieur == match.equipe_domicile and m.score_exterieur is not None: buts_dom += m.score_exterieur; nb_dom += 1
        moy_buts_dom = (buts_dom / nb_dom) if nb_dom > 0 else 1.5

        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter((Match.equipe_domicile == match.equipe_exterieur) | (Match.equipe_exterieur == match.equipe_exterieur)).all()
        buts_ext, nb_ext = 0, 0
        for stat, m in stats_ext:
            if m.equipe_domicile == match.equipe_exterieur and m.score_domicile is not None: buts_ext += m.score_domicile; nb_ext += 1
            elif m.equipe_exterieur == match.equipe_exterieur and m.score_exterieur is not None: buts_ext += m.score_exterieur; nb_ext += 1
        moy_buts_ext = (buts_ext / nb_ext) if nb_ext > 0 else 1.2

        # 2. Calcul du temps restant et ajustement des Lambdas
        temps_restant = max(0, 90 - minute)
        facteur_temps = temps_restant / 90 # Proportion du match restante
        
        lambda_dom_restant = moy_buts_dom * facteur_temps
        lambda_ext_restant = moy_buts_ext * facteur_temps

        # 3. Application de Viterbi (État du match en cours)
        etat_viterbi = calc_viterbi_state(minute=minute, score_dom=score_dom, score_ext=score_ext)
        lambda_dom_restant *= etat_viterbi
        lambda_ext_restant *= etat_viterbi

        # 4. Matrice Bivariate Poisson pour le temps restant
        matrice = calc_bivariate_poisson_matrix(lambda_dom_restant, lambda_ext_restant, max_buts=4)
        
        # 5. On cherche la probabilité du score final demandé
        # Ex: Si score actuel = 1-1 et score_pari = "2-1", il faut 1 but de plus pour Dom et 0 pour Ext.
        score_cible = list(map(int, score_pari.split("-")))
        buts_a_marquer_dom = score_cible[0] - score_dom
        buts_a_marquer_ext = score_cible[1] - score_ext

        proba_score_final = 0.0
        if buts_a_marquer_dom >= 0 and buts_a_marquer_ext >= 0:
            # On cherche la proba dans la matrice (probabilité de marquer X buts et Y buts restants)
            for s in matrice:
                i, j = map(int, s["score"].split("-"))
                if i == buts_a_marquer_dom and j == buts_a_marquer_ext:
                    proba_score_final = s["proba"] * 100
                    break
        
        # 6. Application de la Couche 6 (Market Drift)
        drift_result = calc_live_score_drift(proba_score_final, cote_score)

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "etat_actuel": {
                "minute": minute,
                "score_actuel": f"{score_dom} - {score_ext}",
                "score_pari": score_pari,
                "cote_bookmaker": cote_score
            },
            "analyse_live_score": {
                "probabilite_estimee_par_orion": round(proba_score_final, 2)
            },
            "verdict_market_drift": drift_result
        }
    except Exception as e: return {"erreur": str(e)}
    finally: db.close()

@app.get("/predict-ensemble-cards/{match_id_local}")
async def predict_ensemble_cards(match_id_local: int, arbitre_profil: str = "Normal", cote: float = 1.85):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # 1. Moyenne du championnat (Cartons)
        tous_les_matchs = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.competition_id == match.competition_id).all()
        total_cartons_champ, nb_matchs_champ = 0, 0
        for stat, m in tous_les_matchs:
            if stat.cartons_jaunes_domicile is not None and stat.cartons_jaunes_exterieur is not None:
                total_cartons_champ += (stat.cartons_jaunes_domicile + stat.cartons_jaunes_exterieur)
                nb_matchs_champ += 1
        moy_championnat = (total_cartons_champ / nb_matchs_champ) if nb_matchs_champ > 0 else 4.5

        # 2. Historiques
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_domicile == match.equipe_domicile).all()
        hist_dom = []
        total_cartons_dom, nb_dom = 0, 0
        for stat, m in stats_dom:
            if stat.cartons_jaunes_domicile is not None:
                hist_dom.append({"value": stat.cartons_jaunes_domicile, "date": m.date_heure})
                total_cartons_dom += stat.cartons_jaunes_domicile
                nb_dom += 1
        
        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_exterieur == match.equipe_exterieur).all()
        hist_ext = []
        total_cartons_ext, nb_ext = 0, 0
        for stat, m in stats_ext:
            if stat.cartons_jaunes_exterieur is not None:
                hist_ext.append({"value": stat.cartons_jaunes_exterieur, "date": m.date_heure})
                total_cartons_ext += stat.cartons_jaunes_exterieur
                nb_ext += 1

        # 3. Impact de l'arbitre
        impact_arbitre = 1.0
        if arbitre_profil == "Sévère": impact_arbitre = 1.20
        elif arbitre_profil == "Laxiste": impact_arbitre = 0.85

        # 4. Application des Lambdas
        lambda_dc_total = ((calc_lambda_dixon_coles([{'corners': h['value'], 'date': h['date']} for h in hist_dom]) if nb_dom > 0 else moy_championnat / 2) + 
                           (calc_lambda_dixon_coles([{'corners': h['value'], 'date': h['date']} for h in hist_ext]) if nb_ext > 0 else moy_championnat / 2)) * impact_arbitre
                           
        lambda_bayes_total = (calc_lambda_bayesien(total_cartons_dom, nb_dom, moy_championnat) + calc_lambda_bayesien(total_cartons_ext, nb_ext, moy_championnat)) * impact_arbitre
        lambda_simple_total = (((total_cartons_dom / nb_dom) if nb_dom > 0 else moy_championnat / 2) + ((total_cartons_ext / nb_ext) if nb_ext > 0 else moy_championnat / 2)) * impact_arbitre

        # 5. Modèles Aliens
        liste_vals_dom = [h['value'] for h in hist_dom]
        liste_vals_ext = [h['value'] for h in hist_ext]
        
        impact_hawkes_total = (calc_hawkes_impact(liste_vals_dom) + calc_hawkes_impact(liste_vals_ext)) / 2
        lambda_hawkes_total = lambda_simple_total * impact_hawkes_total

        lambda_kalman_total = ((calc_kalman_filter(liste_vals_dom) if nb_dom > 0 else moy_championnat / 2) + (calc_kalman_filter(liste_vals_ext) if nb_ext > 0 else moy_championnat / 2)) * impact_arbitre

        impact_markov_dom = calc_markov_impact(liste_vals_dom)
        impact_markov_ext = calc_markov_impact(liste_vals_ext)
        lambda_markov_total = (((total_cartons_dom / nb_dom) if nb_dom > 0 else moy_championnat / 2) * impact_markov_dom + ((total_cartons_ext / nb_ext) if nb_ext > 0 else moy_championnat / 2) * impact_markov_ext) * impact_arbitre

        # 6. Probabilités (Ligne 4.5)
        ligne = 4.5
        p_poisson = calc_poisson(lambda_simple_total, ligne)
        p_nbinom = calc_binomiale_negative(lambda_simple_total, ligne)
        p_mc = calc_monte_carlo(lambda_simple_total, ligne)
        p_dc = calc_poisson(lambda_dc_total, ligne)
        p_bayes = calc_poisson(lambda_bayes_total, ligne)
        p_hawkes = calc_poisson(lambda_hawkes_total, ligne)
        p_kalman = calc_poisson(lambda_kalman_total, ligne)
        p_markov = calc_poisson(lambda_markov_total, ligne)

        # 7. Ensemble
        liste_probas = [p_poisson, p_nbinom, p_mc, p_dc, p_bayes, p_hawkes, p_kalman, p_markov]
        moyenne_ensemble = np.mean(liste_probas)
        ecart_type = np.std(liste_probas)

        modeles_proches = sum(1 for p in liste_probas if abs(p - moyenne_ensemble) <= 10)
        proba_calibree_cards = calibrer_probabilite(moyenne_ensemble, ecart_type, modeles_proches)
        if modeles_proches >= 7: fiabilite = "HAUTE ✅"
        elif modeles_proches >= 5: fiabilite = "MOYENNE ⚠️"
        else: fiabilite = "FAIBLE ❌ (Divergence)"

        shadow_result = calc_shadow_market(moyenne_ensemble, cote)

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "marche": "Cartons (+4.5)",
            "proba_calibree": round(proba_calibree_cards, 2),
            "moyennes_attendues": {
                "simple": round(lambda_simple_total, 2),
                "dixon_coles": round(lambda_dc_total, 2),
                "bayesien": round(lambda_bayes_total, 2),
                "hawkes": round(lambda_hawkes_total, 2),
                "kalman": round(lambda_kalman_total, 2),
                "markov": round(lambda_markov_total, 2)
            },
            "probabilites": {
                "poisson": round(p_poisson, 2),
                "binomiale_negative": round(p_nbinom, 2),
                "monte_carlo": round(p_mc, 2),
                "dixon_coles": round(p_dc, 2),
                "bayesien": round(p_bayes, 2),
                "hawkes": round(p_hawkes, 2),
                "kalman": round(p_kalman, 2),
                "markov": round(p_markov, 2)
            },
            "ensemble_final": round(moyenne_ensemble, 2),
            "proba_calibree": round(proba_calibree_cards, 2), # <--- NOUVELLE LIGNE
            "ecart_type": round(ecart_type, 2),
            "fiabilite": fiabilite,
            "shadow_market": shadow_result
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/predict-live-cards/{match_id_local}")
async def predict_live_cards(
    match_id_local: int, 
    minute: int, 
    cartons_dom: int, 
    cartons_ext: int, 
    score_dom: int, 
    score_ext: int, 
    ligne: float, 
    cote: float,
    arbitre_profil: str = "Normal"
):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # 1. Récupération des moyennes historiques de cartons
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_domicile == match.equipe_domicile).all()
        c_dom, nb_dom = 0, 0
        for stat, m in stats_dom:
            if stat.cartons_jaunes_domicile is not None:
                c_dom += stat.cartons_jaunes_domicile; nb_dom += 1
        moy_dom = (c_dom / nb_dom) if nb_dom > 0 else 2.0

        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_exterieur == match.equipe_exterieur).all()
        c_ext, nb_ext = 0, 0
        for stat, m in stats_ext:
            if stat.cartons_jaunes_exterieur is not None:
                c_ext += stat.cartons_jaunes_exterieur; nb_ext += 1
        moy_ext = (c_ext / nb_ext) if nb_ext > 0 else 2.0

        # 2. Calcul Live
        cartons_actuels_total = cartons_dom + cartons_ext
        resultat_live = calc_live_cards(
            moyenne_cartons_dom=moy_dom, 
            moyenne_cartons_ext=moy_ext, 
            minute_actuelle=minute, 
            cartons_actuels=cartons_actuels_total, 
            score_dom=score_dom, 
            score_ext=score_ext, 
            arbitre_profil=arbitre_profil,
            ligne=ligne
        )

        # 3. Calcul de la Value (EV)
        proba_decimale = resultat_live["probabilite_live"] / 100
        ev = (proba_decimale * cote) - 1
        
        verdict = f"🔥 VALUE BET LIVE (+{round(ev*100, 1)}%)" if ev > 0 else f"❌ Pas de Value ({round(ev*100, 1)}%)"

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "etat_actuel": {
                "minute": minute,
                "score": f"{score_dom} - {score_ext}",
                "cartons_actuels": cartons_actuels_total,
                "ligne_visee": ligne,
                "cote_saisie": cote,
                "arbitre": arbitre_profil
            },
            "analyse_live": resultat_live,
            "verdict_value": verdict
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.post("/ajouter-resultat/{prediction_id}")
async def ajouter_resultat(prediction_id: int, resultat_reel: float):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not pred: return {"erreur": "Prédiction introuvable."}

        # On sécurise la ligne de référence au cas où elle serait vide
        ligne_reference = pred.ligne if pred.ligne is not None else (9.5 if pred.marche == 'corners' else 4.5)

        # Mise à jour du résultat et du statut
        pred.resultat_reel = resultat_reel
        pred.date_resultat = datetime.utcnow()
        
        if resultat_reel > ligne_reference:
            pred.statut_prediction = "gagne"
        else:
            pred.statut_prediction = "perdu"

        db.commit()
        return {"message": f"Résultat enregistré ! Statut : {pred.statut_prediction.upper()}"}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/predict-advanced/{match_id_local}")
async def predict_advanced(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # 1. Calcul des moyennes de base
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_domicile == match.equipe_domicile).all()
        c_dom, nb_dom = 0, 0
        liste_dom = []
        for stat, m in stats_dom:
            if stat.corners_domicile is not None:
                c_dom += stat.corners_domicile; nb_dom += 1
                liste_dom.append(stat.corners_domicile)
        
        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_exterieur == match.equipe_exterieur).all()
        c_ext, nb_ext = 0, 0
        liste_ext = []
        for stat, m in stats_ext:
            if stat.corners_exterieur is not None:
                c_ext += stat.corners_exterieur; nb_ext += 1
                liste_ext.append(stat.corners_exterieur)
                
        moy_dom = (c_dom / nb_dom) if nb_dom > 0 else 5.5
        moy_ext = (c_ext / nb_ext) if nb_ext > 0 else 4.0
        lambda_simple = moy_dom + moy_ext
        ligne = 9.5

        # 2. Application des 5 innovations
        # Innovation 1 : ZIP (Zero-Inflated)
        p_zip = calc_zero_inflated_poisson(lambda_simple, ligne, zero_prob=0.05)
        
        # Innovation 2 : MMPP (Markov-Modulated) basé sur l'état à la 45ème
        etat = calc_viterbi_state(minute=45, score_dom=0, score_ext=0)
        lambda_mmpp = calc_mmpp(lambda_simple, etat)
        p_mmpp = calc_poisson(lambda_mmpp, ligne)
        
        # Innovation 3 : PAR (Auto-Régressif) simulé sur les 10 dernières minutes (approx)
        # On simule un historique récent avec la fin de la liste
        recent_corners = liste_dom[-3:] + liste_ext[-3:] 
        p_par, lambda_par = calc_live_par(recent_corners, temps_restant=10, ligne=2.5)

        # Innovation 4 : Hamilton (Régime)
        mult_ham_dom, regime_dom = calc_hamilton_regime(liste_dom)
        mult_ham_ext, regime_ext = calc_hamilton_regime(liste_ext)
        lambda_hamilton = (moy_dom * mult_ham_dom) + (moy_ext * mult_ham_ext)
        p_hamilton = calc_poisson(lambda_hamilton, ligne)

        # Innovation 5 : Hurst (Fractales)
        hurst_val, statut_hurst = calc_hurst_exponent(liste_dom + liste_ext)
        # Si Hot Streak (H > 0.6), on augmente un peu la proba
        p_hurst = calc_poisson(lambda_simple * (1 + (hurst_val - 0.5)), ligne)

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "lambda_base": round(lambda_simple, 2),
            "innovations": {
                "1_zero_inflated_poisson": round(p_zip, 2),
                "2_mmpp": {
                    "lambda_ajuste": round(lambda_mmpp, 2),
                    "probabilite": round(p_mmpp, 2)
                },
                "3_par_live": {
                    "lambda_local": round(lambda_par, 2),
                    "probabilite_10min": round(p_par, 2)
                },
                "4_hamilton_regime": {
                    "regime_domicile": regime_dom,
                    "regime_exterieur": regime_ext,
                    "lambda_ajuste": round(lambda_hamilton, 2),
                    "probabilite": round(p_hamilton, 2)
                },
                "5_hurst": {
                    "valeur_H": round(hurst_val, 3),
                    "statut": statut_hurst,
                    "probabilite": round(p_hurst, 2)
                }
            }
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/test-ucl")
async def test_ucl():
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key: return {"erreur": "Clé API introuvable"}
    
    # On demande les 10 prochains matchs de LDC (ID 2)
    url = "https://v3.football.api-sports.io/fixtures?league=2&season=2026&next=10"
    headers = {"x-apisports-key": api_key}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        
    return response.json()

@app.get("/debug-date/{date_test}")
async def debug_date(date_test: str):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key: return {"erreur": "Clé API introuvable"}
    
    url = f"https://v3.football.api-sports.io/fixtures?date={date_test}"
    headers = {"x-apisports-key": api_key}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    data = response.json()
    
    total = data.get("results", 0)
    
    premiers_matchs = []
    if total > 0:
        for m in data["response"][:5]: # On affiche les 5 premiers
            premiers_matchs.append({
                "ligue_id": m["league"]["id"],
                "ligue_nom": m["league"]["name"],
                "equipe_dom": m["teams"]["home"]["name"],
                "equipe_ext": m["teams"]["away"]["name"]
            })
            
    return {
        "date_demandee": date_test,
        "total_matchs_trouves": total,
        "exemples_de_matchs": premiers_matchs
    }

@app.get("/scan-today")
async def scan_today():
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        # On prend les 50 derniers matchs de la base
        matchs_du_jour = db.query(Match).order_by(Match.id.desc()).limit(50).all()
        if not matchs_du_jour: return {"message": "Aucun match dans la base."}

        opportunites = []

        for match in matchs_du_jour:
            # ⚠️ LIGNE COMMENTÉE POUR LE TEST : On accepte les matchs terminés
            # if match.statut == "FT": continue
                
            try:
                res = await predict_ensemble(match.id, meteo="Sec", cote=1.85)
                if "erreur" in res: continue
                    
                if "HAUTE" in res["fiabilite"] or "MOYENNE" in res["fiabilite"]:
                    opportunites.append({
                        "match_id": match.id,
                        "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
                        "marche": "Corners (+9.5)",
                        "probabilite": res["ensemble_final"],
                        "fiabilite": res["fiabilite"],
                        "ecart_type": res["ecart_type"]
                    })
            except Exception as e:
                continue

        opportunites.sort(key=lambda x: x["probabilite"], reverse=True)
        top_3 = opportunites[:3]

        if not top_3: return {"message": "Aucun match avec une fiabilité suffisante n'a été trouvé."}

        return {
            "message": f"Scan terminé ! {len(opportunites)} opportunités trouvées.",
            "top_3": top_3
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/sync-today")
async def sync_today():
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key or not SessionLocal: return {"erreur": "Configuration manquante"}
    
    db = SessionLocal()
    try:
        dates_recherche = [datetime.now().strftime("%Y-%m-%d"), (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")]
        top_leagues = [{"id": 39, "season": 2026}, {"id": 140, "season": 2026}, {"id": 135, "season": 2026}, {"id": 61, "season": 2026}, {"id": 78, "season": 2026}]
        total_matchs_ajoutes = 0

        for date_recherche in dates_recherche:
            for ligue in top_leagues:
                url = f"https://v3.football.api-sports.io/fixtures?date={date_recherche}&league={ligue['id']}&season={ligue['season']}"
                headers = {"x-apisports-key": api_key}
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, headers=headers)
                data = response.json()
                
                if data.get("results", 0) == 0: continue

                for m in data["response"]:
                    id_match_api = m["fixture"]["id"]
                    competition_locale = db.query(Competition).filter(Competition.id_source == ligue["id"]).first()
                    
                    if competition_locale:
                        match_local = db.query(Match).filter(Match.id_source_apifootball == id_match_api).first()
                        if not match_local:
                            date_str = m["fixture"]["date"].replace("Z", "+00:00")
                            nouveau_match = Match(
                                competition_id=competition_locale.id,
                                date_heure=datetime.fromisoformat(date_str),
                                equipe_domicile=m["teams"]["home"]["name"],
                                equipe_exterieur=m["teams"]["away"]["name"],
                                statut=m["fixture"]["status"]["short"],
                                id_source_apifootball=id_match_api,
                                score_domicile=m["goals"]["home"],
                                score_exterieur=m["goals"]["away"],
                                logo_domicile=m["teams"]["home"]["logo"],
                                logo_exterieur=m["teams"]["away"]["logo"]
                            )
                            db.add(nouveau_match)
                            total_matchs_ajoutes += 1
        
        db.commit()
        return {"message": f"Succès ! {total_matchs_ajoutes} matchs ajoutés via l'API (recherche par ligue)."}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/test-5dollar")
async def test_5dollar():
    api_key = os.getenv("FIVE_DOLLAR_KEY")
    if not api_key:
        return {"erreur": "Clé 5Dollar introuvable"}
    
    # On calcule les timestamps Unix pour aujourd'hui (00:00 UTC) et demain (00:00 UTC)
    aujourd_hui = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    demain = aujourd_hui + timedelta(days=1)
    
    start_ts = int(aujourd_hui.timestamp())
    end_ts = int(demain.timestamp())
    
    url = f"https://api.5dollarfootballapi.com/v1/fixtures?start_time={start_ts}&end_time={end_ts}"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            
        if response.status_code != 200:
            return {"erreur": f"Code {response.status_code}", "details": response.text}
            
        data = response.json()
        
        return {
            "status": "Succès !",
            "start_ts": start_ts,
            "end_ts": end_ts,
            "nombre_de_matchs": len(data.get("data", [])),
            "extrait": data.get("data", [])[:2]
        }
    except Exception as e:
        return {"erreur": str(e)}

@app.get("/sync-stats-today")
async def sync_stats_today(days_ago: int = 1):
    api_key = os.getenv("FIVE_DOLLAR_KEY")
    if not api_key or not SessionLocal: return {"erreur": "Configuration manquante"}
    
    db = SessionLocal()
    try:
        # Date cible (par défaut hier, ou il y a X jours)
        target_date = datetime.now() - timedelta(days=days_ago)
        debut = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        fin = debut + timedelta(days=1)
        
        start_ts = int(debut.timestamp())
        end_ts = int(fin.timestamp())
        
        url_matchs = f"https://api.5dollarfootballapi.com/v1/fixtures?start_time={start_ts}&end_time={end_ts}"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url_matchs, headers=headers)
            
        if response.status_code != 200:
            return {"erreur": f"Erreur API : {response.status_code}"}
            
        data = response.json()
        matchs_api = data.get("data", [])
        
        if not matchs_api:
            return {"message": f"Aucun match trouvé dans l'API pour il y a {days_ago} jour(s)."}

        total_stats_ajoutees = 0

        for m in matchs_api:
            if m.get("status") != "finished":
                continue
                
            id_match_api = m["id"]
            
            # 1. On cherche le match ou on le crée s'il n'existe pas
            match_local = db.query(Match).filter(Match.id_source_apifootball == id_match_api).first()
            
            if not match_local:
                nom_ligue_api = m["league"]["name"].lower()
                competition_locale = None
                for comp in db.query(Competition).all():
                    if comp.nom.lower() in nom_ligue_api:
                        competition_locale = comp
                        break
                
                if not competition_locale:
                    continue # On ignore si on ne connait pas la compétition
                    
                date_str = m["kickoff_utc"].replace("Z", "+00:00")
                match_local = Match(
                    competition_id=competition_locale.id,
                    date_heure=datetime.fromisoformat(date_str),
                    equipe_domicile=m["teams"]["home"]["name"],
                    equipe_exterieur=m["teams"]["away"]["name"],
                    statut="finished",
                    id_source_apifootball=id_match_api,
                    score_domicile=m.get("goals", {}).get("home"),
                    score_exterieur=m.get("goals", {}).get("away")
                )
                db.add(match_local)
                db.commit()
                db.refresh(match_local)
                
            # 2. On vérifie si on a déjà les stats
            deja_stats = db.query(MatchStat).filter(MatchStat.match_id == match_local.id).first()
            if deja_stats:
                continue
            
            # 3. On appelle l'API pour les stats
            url_stats = f"https://api.5dollarfootballapi.com/v1/fixtures/{id_match_api}/statistics"
            async with httpx.AsyncClient() as client:
                res_stats = await client.get(url_stats, headers=headers)
                
            if res_stats.status_code == 200:
                data_stats = res_stats.json()
                stats = data_stats.get("data", {}).get("statistics", {})
                
                if not stats: continue

                tirs_cadres_dom = stats.get("shots_on_target", {}).get("home", 0) or 0
                tirs_cadres_ext = stats.get("shots_on_target", {}).get("away", 0) or 0
                tirs_hors_dom = stats.get("shots_off_target", {}).get("home", 0) or 0
                tirs_hors_ext = stats.get("shots_off_target", {}).get("away", 0) or 0
                
                total_tirs_dom = tirs_cadres_dom + tirs_hors_dom
                total_tirs_ext = tirs_cadres_ext + tirs_hors_ext
                
                attaques_dang_dom = stats.get("dangerous_attacks", {}).get("home", 0) or 0
                attaques_dang_ext = stats.get("dangerous_attacks", {}).get("away", 0) or 0

                corners_dom = stats.get("corners", {}).get("home", 0) or 0
                corners_ext = stats.get("corners", {}).get("away", 0) or 0
                jaunes_dom = stats.get("yellow_cards", {}).get("home", 0) or 0
                jaunes_ext = stats.get("yellow_cards", {}).get("away", 0) or 0

                nouvelle_stat = MatchStat(
                    match_id=match_local.id,
                    source_api="5dollar",
                    corners_domicile=corners_dom,
                    corners_exterieur=corners_ext,
                    cartons_jaunes_domicile=jaunes_dom,
                    cartons_jaunes_exterieur=jaunes_ext,
                    tirs_domicile=total_tirs_dom,
                    tirs_exterieur=total_tirs_ext,
                    tirs_cadres_domicile=tirs_cadres_dom,
                    tirs_cadres_exterieur=tirs_cadres_ext,
                    attaques_dangereuses_domicile=attaques_dang_dom,
                    attaques_dangereuses_exterieur=attaques_dang_ext,
                    statut_validation="confirmé"
                )
                db.add(nouvelle_stat)
                total_stats_ajoutees += 1
        
        db.commit()
        return {"message": f"Succès ! {total_stats_ajoutees} fiches ajoutées pour il y a {days_ago} jour(s). (Matchs analysés: {len(matchs_api)})"}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/test-stats-5dollar/{match_id_api}")
async def test_stats_5dollar(match_id_api: int):
    api_key = os.getenv("FIVE_DOLLAR_KEY")
    if not api_key: return {"erreur": "Clé manquante"}
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Méthode 1 : On essaie avec ?id= au lieu de ?fixture_id=
    url1 = f"https://api.5dollarfootballapi.com/v1/fixtures/statistics?id={match_id_api}"
    async with httpx.AsyncClient() as client:
        res1 = await client.get(url1, headers=headers)
        
    if res1.status_code == 200:
        return {"methode": "?id=", "data": res1.json()}
        
    # Méthode 2 : On essaie avec l'ID directement dans l'URL (ex: /fixtures/123/statistics)
    url2 = f"https://api.5dollarfootballapi.com/v1/fixtures/{match_id_api}/statistics"
    async with httpx.AsyncClient() as client:
        res2 = await client.get(url2, headers=headers)
        
    if res2.status_code == 200:
        return {"methode": "/{id}/statistics", "data": res2.json()}
        
    # Si aucune ne marche, on renvoie les deux erreurs pour voir
    return {
        "erreur": "Les deux méthodes ont échoué",
        "erreur_methode_1": res1.text,
        "erreur_methode_2": res2.text
        }

@app.get("/predict-shots/{match_id_local}")
async def predict_shots(match_id_local: int, cote: float = 1.85):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # 1. Moyenne du championnat (pour les tirs)
        tous_les_matchs = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.competition_id == match.competition_id).all()
        total_tirs_champ, nb_matchs_champ = 0, 0
        for stat, m in tous_les_matchs:
            if stat.tirs_domicile is not None and stat.tirs_exterieur is not None:
                total_tirs_champ += (stat.tirs_domicile + stat.tirs_exterieur)
                nb_matchs_champ += 1
        moy_championnat = (total_tirs_champ / nb_matchs_champ) if nb_matchs_champ > 0 else 22.0

        # 2. Historiques des tirs
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_domicile == match.equipe_domicile).all()
        hist_dom = []
        total_tirs_dom, nb_dom = 0, 0
        for stat, m in stats_dom:
            if stat.tirs_domicile is not None:
                hist_dom.append({"corners": stat.tirs_domicile, "date": m.date_heure})
                total_tirs_dom += stat.tirs_domicile
                nb_dom += 1
        
        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_exterieur == match.equipe_exterieur).all()
        hist_ext = []
        total_tirs_ext, nb_ext = 0, 0
        for stat, m in stats_ext:
            if stat.tirs_exterieur is not None:
                hist_ext.append({"corners": stat.tirs_exterieur, "date": m.date_heure})
                total_tirs_ext += stat.tirs_exterieur
                nb_ext += 1

        # 3. Calcul des Lambdas (avec les modèles adaptés)
        lambda_dc_total = (calc_lambda_dixon_coles(hist_dom) if nb_dom > 0 else moy_championnat / 2) + (calc_lambda_dixon_coles(hist_ext) if nb_ext > 0 else moy_championnat / 2)
        lambda_bayes_total = calc_lambda_bayesien(total_tirs_dom, nb_dom, moy_championnat) + calc_lambda_bayesien(total_tirs_ext, nb_ext, moy_championnat)
        lambda_simple_total = ((total_tirs_dom / nb_dom) if nb_dom > 0 else moy_championnat / 2) + ((total_tirs_ext / nb_ext) if nb_ext > 0 else moy_championnat / 2)

        liste_tirs_dom = [h['corners'] for h in hist_dom]
        liste_tirs_ext = [h['corners'] for h in hist_ext]
        
        impact_hawkes_total = (calc_hawkes_impact(liste_tirs_dom) + calc_hawkes_impact(liste_tirs_ext)) / 2
        lambda_hawkes_total = lambda_simple_total * impact_hawkes_total

        lambda_kalman_total = (calc_kalman_filter(liste_tirs_dom) if nb_dom > 0 else moy_championnat / 2) + (calc_kalman_filter(liste_tirs_ext) if nb_ext > 0 else moy_championnat / 2)

        impact_markov_dom = calc_markov_impact(liste_tirs_dom)
        impact_markov_ext = calc_markov_impact(liste_tirs_ext)
        lambda_markov_total = (((total_tirs_dom / nb_dom) if nb_dom > 0 else moy_championnat / 2) * impact_markov_dom + ((total_tirs_ext / nb_ext) if nb_ext > 0 else moy_championnat / 2) * impact_markov_ext)

        # Innovations
        ligne = 22.5 # Ligne classique pour les tirs totaux
        
        p_zip = calc_zero_inflated_poisson(lambda_simple_total, ligne, zero_prob=0.05)
        
        etat = calc_viterbi_state(minute=45, score_dom=0, score_ext=0)
        lambda_mmpp = calc_mmpp(lambda_simple_total, etat)
        p_mmpp = calc_poisson(lambda_mmpp, ligne)

        recent_tirs = liste_tirs_dom[-3:] + liste_tirs_ext[-3:] 
        p_par, _ = calc_live_par(recent_tirs, temps_restant=10, ligne=5.5)

        mult_ham_dom, _ = calc_hamilton_regime(liste_tirs_dom)
        mult_ham_ext, _ = calc_hamilton_regime(liste_tirs_ext)
        lambda_hamilton = ((total_tirs_dom / nb_dom) * mult_ham_dom + (total_tirs_ext / nb_ext) * mult_ham_ext) if nb_dom > 0 and nb_ext > 0 else lambda_simple_total
        p_hamilton = calc_poisson(lambda_hamilton, ligne)

        hurst_val, _ = calc_hurst_exponent(liste_tirs_dom + liste_tirs_ext)
        p_hurst = calc_poisson(lambda_simple_total * (1 + (hurst_val - 0.5)), ligne)

        # Modèles de base
        p_poisson = calc_poisson(lambda_simple_total, ligne)
        p_nbinom = calc_binomiale_negative(lambda_simple_total, ligne)
        p_mc = calc_monte_carlo(lambda_simple_total, ligne)
        p_dc = calc_poisson(lambda_dc_total, ligne)
        p_bayes = calc_poisson(lambda_bayes_total, ligne)
        p_hawkes = calc_poisson(lambda_hawkes_total, ligne)
        p_kalman = calc_poisson(lambda_kalman_total, ligne)
        p_markov = calc_poisson(lambda_markov_total, ligne)

        # Ensemble final (13 modèles)
        liste_probas = [p_poisson, p_nbinom, p_mc, p_dc, p_bayes, p_hawkes, p_kalman, p_markov, p_zip, p_mmpp, p_par, p_hamilton, p_hurst]
        poids = [1]*8 + [1.5]*5
        moyenne_ensemble = np.average(liste_probas, weights=poids)
        ecart_type = np.std(liste_probas)

        modeles_proches = sum(1 for p in liste_probas if abs(p - moyenne_ensemble) <= 10)
        proba_calibree_shots = calibrer_probabilite(moyenne_ensemble, ecart_type, modeles_proches)
        if modeles_proches >= 10: fiabilite = "HAUTE ✅"
        elif modeles_proches >= 7: fiabilite = "MOYENNE ⚠️"
        else: fiabilite = "FAIBLE ❌ (Divergence)"

        shadow_result = calc_shadow_market(moyenne_ensemble, cote)

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "marche": "Tirs Totaux (+22.5)",
            "ensemble_final": round(moyenne_ensemble, 2),
            "ensemble_final": round(moyenne_ensemble, 2),
            "proba_calibree": round(proba_calibree_shots, 2), # <--- NOUVELLE LIGNE
            "fiabilite": fiabilite,
            "shadow_market": shadow_result
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/predict-shots-target/{match_id_local}")
async def predict_shots_target(match_id_local: int, cote: float = 1.85):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # 1. Moyenne du championnat (Tirs Cadrés)
        tous_les_matchs = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.competition_id == match.competition_id).all()
        total_champ, nb_champ = 0, 0
        for stat, m in tous_les_matchs:
            if stat.tirs_cadres_domicile is not None and stat.tirs_cadres_exterieur is not None:
                total_champ += (stat.tirs_cadres_domicile + stat.tirs_cadres_exterieur)
                nb_champ += 1
        moy_championnat = (total_champ / nb_champ) if nb_champ > 0 else 9.0 # Moyenne par défaut

        # 2. Historiques
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_domicile == match.equipe_domicile).all()
        hist_dom = []
        total_dom, nb_dom = 0, 0
        for stat, m in stats_dom:
            if stat.tirs_cadres_domicile is not None:
                hist_dom.append({"corners": stat.tirs_cadres_domicile, "date": m.date_heure})
                total_dom += stat.tirs_cadres_domicile
                nb_dom += 1
        
        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_exterieur == match.equipe_exterieur).all()
        hist_ext = []
        total_ext, nb_ext = 0, 0
        for stat, m in stats_ext:
            if stat.tirs_cadres_exterieur is not None:
                hist_ext.append({"corners": stat.tirs_cadres_exterieur, "date": m.date_heure})
                total_ext += stat.tirs_cadres_exterieur
                nb_ext += 1

        # 3. Calcul des Lambdas
        lambda_dc_total = (calc_lambda_dixon_coles(hist_dom) if nb_dom > 0 else moy_championnat / 2) + (calc_lambda_dixon_coles(hist_ext) if nb_ext > 0 else moy_championnat / 2)
        lambda_bayes_total = calc_lambda_bayesien(total_dom, nb_dom, moy_championnat) + calc_lambda_bayesien(total_ext, nb_ext, moy_championnat)
        lambda_simple_total = ((total_dom / nb_dom) if nb_dom > 0 else moy_championnat / 2) + ((total_ext / nb_ext) if nb_ext > 0 else moy_championnat / 2)

        liste_dom = [h['corners'] for h in hist_dom]
        liste_ext = [h['corners'] for h in hist_ext]
        
        impact_hawkes_total = (calc_hawkes_impact(liste_dom) + calc_hawkes_impact(liste_ext)) / 2
        lambda_hawkes_total = lambda_simple_total * impact_hawkes_total

        lambda_kalman_total = (calc_kalman_filter(liste_dom) if nb_dom > 0 else moy_championnat / 2) + (calc_kalman_filter(liste_ext) if nb_ext > 0 else moy_championnat / 2)

        impact_markov_dom = calc_markov_impact(liste_dom)
        impact_markov_ext = calc_markov_impact(liste_ext)
        lambda_markov_total = (((total_dom / nb_dom) if nb_dom > 0 else moy_championnat / 2) * impact_markov_dom + ((total_ext / nb_ext) if nb_ext > 0 else moy_championnat / 2) * impact_markov_ext)

        # Innovations
        ligne = 8.5 # Ligne classique pour les tirs cadrés
        
        p_zip = calc_zero_inflated_poisson(lambda_simple_total, ligne, zero_prob=0.05)
        
        etat = calc_viterbi_state(minute=45, score_dom=0, score_ext=0)
        lambda_mmpp = calc_mmpp(lambda_simple_total, etat)
        p_mmpp = calc_poisson(lambda_mmpp, ligne)

        recent = liste_dom[-3:] + liste_ext[-3:] 
        p_par, _ = calc_live_par(recent, temps_restant=10, ligne=2.5)

        mult_ham_dom, _ = calc_hamilton_regime(liste_dom)
        mult_ham_ext, _ = calc_hamilton_regime(liste_ext)
        lambda_hamilton = ((total_dom / nb_dom) * mult_ham_dom + (total_ext / nb_ext) * mult_ham_ext) if nb_dom > 0 and nb_ext > 0 else lambda_simple_total
        p_hamilton = calc_poisson(lambda_hamilton, ligne)

        hurst_val, _ = calc_hurst_exponent(liste_dom + liste_ext)
        p_hurst = calc_poisson(lambda_simple_total * (1 + (hurst_val - 0.5)), ligne)

        # Modèles de base
        p_poisson = calc_poisson(lambda_simple_total, ligne)
        p_nbinom = calc_binomiale_negative(lambda_simple_total, ligne)
        p_mc = calc_monte_carlo(lambda_simple_total, ligne)
        p_dc = calc_poisson(lambda_dc_total, ligne)
        p_bayes = calc_poisson(lambda_bayes_total, ligne)
        p_hawkes = calc_poisson(lambda_hawkes_total, ligne)
        p_kalman = calc_poisson(lambda_kalman_total, ligne)
        p_markov = calc_poisson(lambda_markov_total, ligne)

        # Ensemble final (13 modèles)
        liste_probas = [p_poisson, p_nbinom, p_mc, p_dc, p_bayes, p_hawkes, p_kalman, p_markov, p_zip, p_mmpp, p_par, p_hamilton, p_hurst]
        poids = [1]*8 + [1.5]*5
        moyenne_ensemble = np.average(liste_probas, weights=poids)
        ecart_type = np.std(liste_probas)

        modeles_proches = sum(1 for p in liste_probas if abs(p - moyenne_ensemble) <= 10)
        proba_calibree_shots_target = calibrer_probabilite(moyenne_ensemble, ecart_type, modeles_proches)
        if modeles_proches >= 10: fiabilite = "HAUTE ✅"
        elif modeles_proches >= 7: fiabilite = "MOYENNE ⚠️"
        else: fiabilite = "FAIBLE ❌ (Divergence)"

        shadow_result = calc_shadow_market(moyenne_ensemble, cote)

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "marche": "Tirs Cadrés (+8.5)",
            "ensemble_final": round(moyenne_ensemble, 2),
            "ensemble_final": round(moyenne_ensemble, 2),
            "proba_calibree": round(proba_calibree_shots_target, 2), # <--- NOUVELLE LIGNE
            "fiabilite": fiabilite,
            "shadow_market": shadow_result
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/predict-fautes/{match_id_local}")
async def predict_fautes(match_id_local: int, dom_fautes: float, ext_fautes: float, ligne: float = 25.5, cote: float = 1.85):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        lambda_total = dom_fautes + ext_fautes

        # Application de nos modèles mathématiques
        p_poisson = calc_poisson(lambda_total, ligne)
        p_nbinom = calc_binomiale_negative(lambda_total, ligne)
        p_mc = calc_monte_carlo(lambda_total, ligne)
        p_zip = calc_zero_inflated_poisson(lambda_total, ligne, zero_prob=0.02)
        
        # Modèles dynamiques
        etat_viterbi = calc_viterbi_state(minute=45, score_dom=0, score_ext=0)
        lambda_mmpp = calc_mmpp(lambda_total, etat_viterbi)
        p_mmpp = calc_poisson(lambda_mmpp, ligne)
        
        p_hamilton = calc_poisson(lambda_total * 1.05, ligne)
        p_hurst = calc_poisson(lambda_total * 0.98, ligne)

        liste_probas = [p_poisson, p_nbinom, p_mc, p_zip, p_mmpp, p_hamilton, p_hurst]
        moyenne_ensemble = np.mean(liste_probas)
        borne_basse, borne_haute = calc_intervalle_confiance(liste_probas)
        ecart_type = np.std(liste_probas)

        modeles_proches = sum(1 for p in liste_probas if abs(p - moyenne_ensemble) <= 10)
        
        if modeles_proches >= 5: fiabilite = "HAUTE ✅"
        elif modeles_proches >= 3: fiabilite = "MOYENNE ⚠️"
        else: fiabilite = "FAIBLE ❌ (Divergence)"

        proba_calibree = calibrer_probabilite(moyenne_ensemble, ecart_type, modeles_proches)

        shadow_result = calc_shadow_market(moyenne_ensemble, cote)

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "marche": f"Fautes (+{ligne})",
            "ensemble_final": round(moyenne_ensemble, 2),
            "proba_calibree": round(proba_calibree, 2),
            "fourchette_basse": borne_basse,
            "fourchette_haute": borne_haute,
            "fiabilite": fiabilite,
            "shadow_market": shadow_result
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/predict-touches/{match_id_local}")
async def predict_touches(match_id_local: int, dom_touches: float, ext_touches: float, ligne: float = 950.5, cote: float = 1.85):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        lambda_total = dom_touches + ext_touches

        # Application de nos modèles mathématiques
        p_poisson = calc_poisson(lambda_total, ligne)
        p_nbinom = calc_binomiale_negative(lambda_total, ligne)
        p_mc = calc_monte_carlo(lambda_total, ligne)
        p_zip = calc_zero_inflated_poisson(lambda_total, ligne, zero_prob=0.01)
        
        # Modèles dynamiques
        etat_viterbi = calc_viterbi_state(minute=45, score_dom=0, score_ext=0)
        lambda_mmpp = calc_mmpp(lambda_total, etat_viterbi)
        p_mmpp = calc_poisson(lambda_mmpp, ligne)
        
        p_hamilton = calc_poisson(lambda_total * 1.02, ligne) 
        p_hurst = calc_poisson(lambda_total * 0.98, ligne)

        liste_probas = [p_poisson, p_nbinom, p_mc, p_zip, p_mmpp, p_hamilton, p_hurst]
        moyenne_ensemble = np.mean(liste_probas)
        borne_basse_t, borne_haute_t = calc_intervalle_confiance(liste_probas)
        ecart_type = np.std(liste_probas)

        modeles_proches = sum(1 for p in liste_probas if abs(p - moyenne_ensemble) <= 10)
        
        if modeles_proches >= 5: fiabilite = "HAUTE ✅"
        elif modeles_proches >= 3: fiabilite = "MOYENNE ⚠️"
        else: fiabilite = "FAIBLE ❌ (Divergence)"

        proba_calibree = calibrer_probabilite(moyenne_ensemble, ecart_type, modeles_proches)

        shadow_result = calc_shadow_market(moyenne_ensemble, cote)

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "marche": f"Touches (+{ligne})",
            "ensemble_final": round(moyenne_ensemble, 2),
            "proba_calibree": round(proba_calibree, 2),
            "fourchette_basse": borne_basse_t,
            "fourchette_haute": borne_haute_t, 
            "fiabilite": fiabilite,
            "shadow_market": shadow_result
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()
