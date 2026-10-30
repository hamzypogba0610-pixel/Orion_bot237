from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import httpx
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, Competition, Match, MatchStat, Prediction
from app.engine import calc_poisson, calc_binomiale_negative, calc_monte_carlo, calc_lambda_dixon_coles, calc_lambda_bayesien, calc_hawkes_impact, calc_shadow_market, calc_live_corners
from datetime import datetime
from scipy.stats import poisson
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

        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_domicile) | (Match.equipe_exterieur == match.equipe_domicile)
        ).all()
        buts_dom, nb_matchs_dom = 0, 0
        for stat, m in stats_dom:
            if m.equipe_domicile == match.equipe_domicile and m.score_domicile is not None:
                buts_dom += m.score_domicile
                nb_matchs_dom += 1
            elif m.equipe_exterieur == match.equipe_domicile and m.score_exterieur is not None:
                buts_dom += m.score_exterieur
                nb_matchs_dom += 1
        moyenne_buts_dom = (buts_dom / nb_matchs_dom) if nb_matchs_dom > 0 else 1.5

        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_exterieur) | (Match.equipe_exterieur == match.equipe_exterieur)
        ).all()
        buts_ext, nb_matchs_ext = 0, 0
        for stat, m in stats_ext:
            if m.equipe_domicile == match.equipe_exterieur and m.score_domicile is not None:
                buts_ext += m.score_domicile
                nb_matchs_ext += 1
            elif m.equipe_exterieur == match.equipe_exterieur and m.score_exterieur is not None:
                buts_ext += m.score_exterieur
                nb_matchs_ext += 1
        moyenne_buts_ext = (buts_ext / nb_matchs_ext) if nb_matchs_ext > 0 else 1.2

        lambda_dom_ht = moyenne_buts_dom * 0.45
        lambda_ext_ht = moyenne_buts_ext * 0.45

        scores_possibles = []
        for i in range(4):
            for j in range(4):
                p_dom = poisson.pmf(i, lambda_dom_ht)
                p_ext = poisson.pmf(j, lambda_ext_ht)
                proba = p_dom * p_ext
                scores_possibles.append({"score": f"{i}-{j}", "proba": proba})

        scores_possibles.sort(key=lambda x: x["proba"], reverse=True)
        top_3 = scores_possibles[:3]

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "top_scores_mi_temps": [
                {"score": s["score"], "probabilite": round(s["proba"] * 100, 2)} for s in top_3
            ]
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/import-csv/")
async def import_csv(url: str):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        # 1. Détection du championnat (avec les 10 ligues)
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
        elif "new/BRA.csv" in url: nom_ligue = "Brasileirao"
        elif "new/USA.csv" in url: nom_ligue = "MLS"
        else: return {"erreur": "Lien non reconnu. Vérifie le code du fichier (E0, SP1, I1, F1, D1, N1, P1, B1, T1, G1)."}

        # 2. Recherche de la compétition dans la base
        competition_locale = db.query(Competition).filter(Competition.nom == nom_ligue).first()
        if not competition_locale:
            return {"erreur": f"La compétition {nom_ligue} n'est pas dans la base. Lance /init-competitions d'abord."}
        
        competition_id = competition_locale.id 

        # 3. Téléchargement du fichier CSV
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
        
        if response.status_code != 200: 
            return {"erreur": f"Erreur HTTP : {response.status_code}. Le fichier n'existe peut-être pas."}
            
        csv_file = io.StringIO(response.text)
        reader = csv.DictReader(csv_file)
        
        matchs_ajoutes = 0
        stats_ajoutees = 0

        # 4. Lecture et insertion des données
        for row in reader:
            if not row.get('HomeTeam') or not row.get('AwayTeam') or not row.get('Date'): 
                continue
                
            # Génération d'un ID unique pour éviter les doublons
            fake_id_str = f"{row['HomeTeam']}{row['AwayTeam']}{row['Date']}"
            fake_id = int(hashlib.md5(fake_id_str.encode()).hexdigest()[:8], 16) % 2000000000
            
            if db.query(Match).filter(Match.id_source_apifootball == fake_id).first(): 
                continue

            # Gestion de la date
            try:
                date_obj = datetime.strptime(row['Date'], "%d/%m/%Y")
            except:
                try:
                    date_obj = datetime.strptime(row['Date'], "%d/%m/%y")
                except:
                    date_obj = datetime.now()

            # Création du match
            nouveau_match = Match(
                competition_id=competition_id, date_heure=date_obj,
                equipe_domicile=row['HomeTeam'], equipe_exterieur=row['AwayTeam'], statut="FT",
                id_source_apifootball=fake_id,
                score_domicile=int(row['FTHG']) if row.get('FTHG') and row['FTHG'].isdigit() else None,
                score_exterieur=int(row['FTAG']) if row.get('FTAG') and row['FTAG'].isdigit() else None
            )
            db.add(nouveau_match)
            db.commit()
            db.refresh(nouveau_match)
            matchs_ajoutes += 1

            # Création des statistiques (si disponibles dans le CSV)
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
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key or not SessionLocal: return {"erreur": "Configuration manquante"}
    
    db = SessionLocal()
    try:
        # 1. On demande la date d'aujourd'hui
        aujourd_hui = datetime.now().strftime("%Y-%m-%d")
        url = f"https://v3.football.api-sports.io/fixtures?date={aujourd_hui}"
        headers = {"x-apisports-key": api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
        data = response.json()
        
        if data.get("results", 0) == 0: return {"message": "Aucun match aujourd'hui sur l'API."}

        top_leagues = [39, 140, 135, 61, 78, 88, 94, 144, 203, 197, 2, 3, 71, 307, 253]
        matchs_ajoutes = 0

        for m in data["response"]:
            id_ligue = m["league"]["id"]
            if id_ligue not in top_leagues: continue

            id_match_api = m["fixture"]["id"]
            competition_locale = db.query(Competition).filter(Competition.id_source == id_ligue).first()
            
            if competition_locale:
                # On vérifie si le match existe déjà
                match_local = db.query(Match).filter(Match.id_source_apifootball == id_match_api).first()
                if not match_local:
                    date_str = m["fixture"]["date"].replace("Z", "+00:00")
                    nouveau_match = Match(
                        competition_id=competition_locale.id,
                        date_heure=datetime.fromisoformat(date_str),
                        equipe_domicile=m["teams"]["home"]["name"],
                        equipe_exterieur=m["teams"]["away"]["name"],
                        logo_domicile=m["teams"]["home"]["logo"],
                        logo_exterieur=m["teams"]["away"]["logo"],
                        statut=m["fixture"]["status"]["short"],
                        id_source_apifootball=id_match_api,
                        score_domicile=m["goals"]["home"],
                        score_exterieur=m["goals"]["away"]
                    )
                    db.add(nouveau_match)
                    matchs_ajoutes += 1
        
        db.commit()
        return {"message": f"Succès ! {matchs_ajoutes} matchs du jour ont été ajoutés."}
    except Exception as e:
        db.rollback()
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

        # On enregistre le résultat
        pred.resultat_reel = resultat_reel
        pred.date_resultat = datetime.utcnow()
        
        # On compare avec la ligne (ex: si la ligne est 9.5 et résultat 11, c'est gagné)
        if resultat_reel > pred.ligne:
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

@app.get("/predict-ft-score/{match_id_local}")
async def predict_ft_score(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        # Calcul des moyennes de buts FT (Full Time)
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_domicile) | (Match.equipe_exterieur == match.equipe_domicile)
        ).all()
        buts_dom, nb_matchs_dom = 0, 0
        for stat, m in stats_dom:
            if m.equipe_domicile == match.equipe_domicile and m.score_domicile is not None:
                buts_dom += m.score_domicile; nb_matchs_dom += 1
            elif m.equipe_exterieur == match.equipe_domicile and m.score_exterieur is not None:
                buts_dom += m.score_exterieur; nb_matchs_dom += 1
        moyenne_buts_dom = (buts_dom / nb_matchs_dom) if nb_matchs_dom > 0 else 1.5

        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_exterieur) | (Match.equipe_exterieur == match.equipe_exterieur)
        ).all()
        buts_ext, nb_matchs_ext = 0, 0
        for stat, m in stats_ext:
            if m.equipe_domicile == match.equipe_exterieur and m.score_domicile is not None:
                buts_ext += m.score_domicile; nb_matchs_ext += 1
            elif m.equipe_exterieur == match.equipe_exterieur and m.score_exterieur is not None:
                buts_ext += m.score_exterieur; nb_matchs_ext += 1
        moyenne_buts_ext = (buts_ext / nb_matchs_ext) if nb_matchs_ext > 0 else 1.2

        # Calcul des probabilités pour le score final
        scores_possibles = []
        for i in range(6): # 0 à 5 buts
            for j in range(6):
                p_dom = poisson.pmf(i, moyenne_buts_dom)
                p_ext = poisson.pmf(j, moyenne_buts_ext)
                proba = p_dom * p_ext
                scores_possibles.append({"score": f"{i}-{j}", "proba": proba})

        scores_possibles.sort(key=lambda x: x["proba"], reverse=True)
        top_3 = scores_possibles[:3]

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "top_scores_finaux": [
                {"score": s["score"], "probabilite": round(s["proba"] * 100, 2)} for s in top_3
            ]
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

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

        # 2. Historique Domicile
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_domicile == match.equipe_domicile).all()
        hist_dom = []
        total_corners_dom, nb_dom = 0, 0
        for stat, m in stats_dom:
            if stat.corners_domicile is not None:
                hist_dom.append({"corners": stat.corners_domicile, "date": m.date_heure})
                total_corners_dom += stat.corners_domicile
                nb_dom += 1
        
        # 3. Historique Extérieur
        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(Match.equipe_exterieur == match.equipe_exterieur).all()
        hist_ext = []
        total_corners_ext, nb_ext = 0, 0
        for stat, m in stats_ext:
            if stat.corners_exterieur is not None:
                hist_ext.append({"corners": stat.corners_exterieur, "date": m.date_heure})
                total_corners_ext += stat.corners_exterieur
                nb_ext += 1

        # 4. Calcul des Lambdas
        lambda_dc_dom = calc_lambda_dixon_coles(hist_dom) if nb_dom > 0 else moy_championnat / 2
        lambda_dc_ext = calc_lambda_dixon_coles(hist_ext) if nb_ext > 0 else moy_championnat / 2
        lambda_dc_total = lambda_dc_dom + lambda_dc_ext

        lambda_bayes_dom = calc_lambda_bayesien(total_corners_dom, nb_dom, moy_championnat)
        lambda_bayes_ext = calc_lambda_bayesien(total_corners_ext, nb_ext, moy_championnat)
        lambda_bayes_total = lambda_bayes_dom + lambda_bayes_ext

        lambda_simple_total = ((total_corners_dom / nb_dom) if nb_dom > 0 else moy_championnat / 2) + ((total_corners_ext / nb_ext) if nb_ext > 0 else moy_championnat / 2)

        # Impact Météo
        impact_meteo = 0.90 if meteo == "Pluie" else (0.85 if meteo == "Vent fort" else 1.0)
        lambda_simple_total *= impact_meteo
        lambda_dc_total *= impact_meteo
        lambda_bayes_total *= impact_meteo

        # 5. CALCUL HAWKES
        liste_corners_dom = [h['corners'] for h in hist_dom]
        liste_corners_ext = [h['corners'] for h in hist_ext]
        impact_hawkes_dom = calc_hawkes_impact(liste_corners_dom)
        impact_hawkes_ext = calc_hawkes_impact(liste_corners_ext)
        
        lambda_hawkes_dom = ((total_corners_dom / nb_dom) * impact_hawkes_dom) if nb_dom > 0 else (moy_championnat / 2)
        lambda_hawkes_ext = ((total_corners_ext / nb_ext) * impact_hawkes_ext) if nb_ext > 0 else (moy_championnat / 2)
        lambda_hawkes_total = (lambda_hawkes_dom + lambda_hawkes_ext) * impact_meteo

        # 6. Calcul des probabilités (Ligne 9.5)
        ligne = 9.5
        p_poisson = calc_poisson(lambda_simple_total, ligne)
        p_nbinom = calc_binomiale_negative(lambda_simple_total, ligne)
        p_mc = calc_monte_carlo(lambda_simple_total, ligne)
        p_dc = calc_poisson(lambda_dc_total, ligne)
        p_bayes = calc_poisson(lambda_bayes_total, ligne)
        p_hawkes = calc_poisson(lambda_hawkes_total, ligne)

        # 7. Ensemble et Fiabilité
        liste_probas = [p_poisson, p_nbinom, p_mc, p_dc, p_bayes, p_hawkes]
        moyenne_ensemble = np.mean(liste_probas)
        ecart_type = np.std(liste_probas)

        # NOUVEAU CALCUL DE LA FIABILITÉ
        modeles_proches = sum(1 for p in liste_probas if abs(p - moyenne_ensemble) <= 10)
        
        if modeles_proches >= 6: 
            fiabilite = "HAUTE ✅"
        elif modeles_proches >= 4: 
            fiabilite = "MOYENNE ⚠️"
        else: 
            fiabilite = "FAIBLE ❌ (Divergence)"

        # 8. SHADOW MARKET
        shadow_result = calc_shadow_market(moyenne_ensemble, cote)

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "moyennes_attendues": {
                "simple": round(lambda_simple_total, 2),
                "dixon_coles": round(lambda_dc_total, 2),
                "bayesien": round(lambda_bayes_total, 2),
                "hawkes": round(lambda_hawkes_total, 2)
            },
            "impact_hawkes": {
                "dom": round(impact_hawkes_dom, 2),
                "ext": round(impact_hawkes_ext, 2)
            },
            "probabilites_ligne_9_5": {
                "poisson": round(p_poisson, 2),
                "binomiale_negative": round(p_nbinom, 2),
                "monte_carlo": round(p_mc, 2),
                "dixon_coles": round(p_dc, 2),
                "bayesien": round(p_bayes, 2),
                "hawkes": round(p_hawkes, 2)
            },
            "ensemble_final": round(moyenne_ensemble, 2),
            "nb_modeles_proches": modeles_proches,
            "ecart_type": round(ecart_type, 2),
            "fiabilite": fiabilite,
            "shadow_market": shadow_result
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close() 

@app.get("/scan-today")
async def scan_today():
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        # On prend les 20 derniers matchs ajoutés
        matchs_du_jour = db.query(Match).order_by(Match.id.desc()).limit(20).all()
        if not matchs_du_jour: return {"message": "Aucun match dans la base."}

        opportunites = []

        for match in matchs_du_jour:
            # On ignore les matchs terminés
            if match.statut == "FT": 
                continue
                
            try:
                res = await predict_ensemble(match.id, meteo="Sec", cote=1.85)
                
                if "erreur" in res:
                    continue
                    
                # On accepte HAUTE et MOYENNE
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

        # On trie par probabilité décroissante
        opportunites.sort(key=lambda x: x["probabilite"], reverse=True)
        top_3 = opportunites[:3]

        if not top_3:
            return {"message": "Aucun match avec une fiabilité suffisante n'a été trouvé."}

        return {
            "message": f"Scan terminé ! {len(opportunites)} opportunités trouvées.",
            "top_3": top_3
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
