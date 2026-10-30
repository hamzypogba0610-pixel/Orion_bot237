from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, Competition, Match, MatchStat, Prediction
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
                "date": m.date_heure.strftime("%d/%m/%Y") if m.date_heure else "Date inconnue"
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
                "match": f"{m.equipe_domicile} vs {m.equipe_exterieur}",
                "marche": p.marche,
                "ligne": p.ligne,
                "probabilite": p.probabilite_brute,
                "date": p.date_prediction.strftime("%d/%m/%Y %H:%M") if p.date_prediction else "Date inconnue"
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
        pred_c = Prediction(
            match_id=match_id_local, marche="corners", ligne=9.5,
            probabilite_brute=p_corners, probabilite_calibree=p_corners,
            modele_version="poisson-v1.0", date_prediction=datetime.utcnow(),
            meteo=meteo, style_adversaire=style_adv,
            arbitre=arbitre, arbitre_profil=arbitre_profil
        )
        db.add(pred_c)

        pred_k = Prediction(
            match_id=match_id_local, marche="cartons", ligne=4.5,
            probabilite_brute=p_cartons, probabilite_calibree=p_cartons,
            modele_version="poisson-v1.0", date_prediction=datetime.utcnow(),
            meteo=meteo, style_adversaire=style_adv,
            arbitre=arbitre, arbitre_profil=arbitre_profil
        )
        db.add(pred_k)

        db.commit()
        return {"message": "Analyse et contexte sauvegardés avec succès !"}
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
            {"id_api": 61, "nom": "Ligue 1", "pays": "France"}
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

        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_domicile) | (Match.equipe_exterieur == match.equipe_domicile)
        ).all()
        total_corners_dom, nb_matchs_dom = 0, 0
        for stat, m in stats_dom:
            if m.equipe_domicile == match.equipe_domicile and stat.corners_domicile is not None:
                total_corners_dom += stat.corners_domicile
                nb_matchs_dom += 1
            elif m.equipe_exterieur == match.equipe_domicile and stat.corners_exterieur is not None:
                total_corners_dom += stat.corners_exterieur
                nb_matchs_dom += 1
        moyenne_dom = (total_corners_dom / nb_matchs_dom) if nb_matchs_dom > 0 else 5.5

        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_exterieur) | (Match.equipe_exterieur == match.equipe_exterieur)
        ).all()
        total_corners_ext, nb_matchs_ext = 0, 0
        for stat, m in stats_ext:
            if m.equipe_domicile == match.equipe_exterieur and stat.corners_domicile is not None:
                total_corners_ext += stat.corners_domicile
                nb_matchs_ext += 1
            elif m.equipe_exterieur == match.equipe_exterieur and stat.corners_exterieur is not None:
                total_corners_ext += stat.corners_exterieur
                nb_matchs_ext += 1
        moyenne_ext = (total_corners_ext / nb_matchs_ext) if nb_matchs_ext > 0 else 4.0

        lambda_total = moyenne_dom + moyenne_ext
        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "donnees_utilisees": {"matchs_domicile": nb_matchs_dom, "matchs_exterieur": nb_matchs_ext},
            "analyse_corners": {
                "moyenne_domicile": round(moyenne_dom, 2), "moyenne_exterieur": round(moyenne_ext, 2),
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

        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_domicile) | (Match.equipe_exterieur == match.equipe_domicile)
        ).all()
        total_jaunes_dom, nb_matchs_dom = 0, 0
        for stat, m in stats_dom:
            if m.equipe_domicile == match.equipe_domicile and stat.cartons_jaunes_domicile is not None:
                total_jaunes_dom += stat.cartons_jaunes_domicile
                nb_matchs_dom += 1
            elif m.equipe_exterieur == match.equipe_domicile and stat.cartons_jaunes_exterieur is not None:
                total_jaunes_dom += stat.cartons_jaunes_exterieur
                nb_matchs_dom += 1
        moyenne_dom = (total_jaunes_dom / nb_matchs_dom) if nb_matchs_dom > 0 else 2.0

        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_exterieur) | (Match.equipe_exterieur == match.equipe_exterieur)
        ).all()
        total_jaunes_ext, nb_matchs_ext = 0, 0
        for stat, m in stats_ext:
            if m.equipe_domicile == match.equipe_exterieur and stat.cartons_jaunes_domicile is not None:
                total_jaunes_ext += stat.cartons_jaunes_domicile
                nb_matchs_ext += 1
            elif m.equipe_exterieur == match.equipe_exterieur and stat.cartons_jaunes_exterieur is not None:
                total_jaunes_ext += stat.cartons_jaunes_exterieur
                nb_matchs_ext += 1
        moyenne_ext = (total_jaunes_ext / nb_matchs_ext) if nb_matchs_ext > 0 else 2.0

        lambda_total = moyenne_dom + moyenne_ext
        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "donnees_utilisees": {"matchs_domicile": nb_matchs_dom, "matchs_exterieur": nb_matchs_ext},
            "analyse_cartons": {
                "moyenne_jaunes_domicile": round(moyenne_dom, 2),
                "moyenne_jaunes_exterieur": round(moyenne_ext, 2),
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
        nom_ligue = ""
        if "E0.csv" in url: nom_ligue = "Premier League"
        elif "SP1.csv" in url: nom_ligue = "La Liga"
        elif "I1.csv" in url: nom_ligue = "Serie A"
        elif "F1.csv" in url: nom_ligue = "Ligue 1"
        elif "D1.csv" in url: nom_ligue = "Bundesliga"
        else: return {"erreur": "Lien non reconnu."}

        competition_locale = db.query(Competition).filter(Competition.nom == nom_ligue).first()
        if not competition_locale:
            return {"erreur": f"La compétition {nom_ligue} n'est pas dans la base."}
        
        competition_id = competition_locale.id 

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
        
        if response.status_code != 200: return {"erreur": f"Erreur HTTP : {response.status_code}"}
            
        csv_file = io.StringIO(response.text)
        reader = csv.DictReader(csv_file)
        
        matchs_ajoutes, stats_ajoutees = 0, 0

        for row in reader:
            if not row.get('HomeTeam') or not row.get('AwayTeam') or not row.get('Date'): continue
                
            fake_id_str = f"{row['HomeTeam']}{row['AwayTeam']}{row['Date']}"
            fake_id = int(hashlib.md5(fake_id_str.encode()).hexdigest()[:8], 16) % 2000000000
            
            if db.query(Match).filter(Match.id_source_apifootball == fake_id).first(): continue

            try:
                date_obj = datetime.strptime(row['Date'], "%d/%m/%Y")
            except:
                try:
                    date_obj = datetime.strptime(row['Date'], "%d/%m/%y")
                except:
                    date_obj = datetime.now()

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
