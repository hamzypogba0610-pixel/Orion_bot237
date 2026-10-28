from fastapi import FastAPI
import os
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, Competition, Match, MatchStat
from datetime import datetime

app = FastAPI(title="ORION", version="0.1.0")

# --- CONNEXION À LA BASE DE DONNÉES ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None

if engine:
    Base.metadata.create_all(bind=engine)

@app.get("/")
async def home():
    return {"message": "ORION est en ligne", "version": "0.1.0", "statut": "vivant"}

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

@app.get("/sync-matches/{date_recherche}")
async def sync_matches(date_recherche: str):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key or not SessionLocal: return {"erreur": "Configuration manquante"}
    url = f"https://v3.football.api-sports.io/fixtures?date={date_recherche}"
    headers = {"x-apisports-key": api_key}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    data = response.json()
    if data.get("results", 0) == 0: return {"message": "Aucun match ce jour-là."}

    db = SessionLocal()
    try:
        ajoutes = 0
        for m in data["response"]:
            competition_locale = db.query(Competition).filter(Competition.id_source == m["league"]["id"]).first()
            if competition_locale:
                id_match_api = m["fixture"]["id"]
                existe = db.query(Match).filter(Match.id_source_apifootball == id_match_api).first()
                if not existe:
                    date_str = m["fixture"]["date"].replace("Z", "+00:00")
                    db.add(Match(
                        competition_id=competition_locale.id,
                        date_heure=datetime.fromisoformat(date_str),
                        equipe_domicile=m["teams"]["home"]["name"],
                        equipe_exterieur=m["teams"]["away"]["name"],
                        statut=m["fixture"]["status"]["short"],
                        id_source_apifootball=id_match_api,
                        score_domicile=m["goals"]["home"],
                        score_exterieur=m["goals"]["away"]
                    ))
                    ajoutes += 1
        db.commit()
        return {"message": f"Succès ! {ajoutes} matchs ajoutés pour le {date_recherche}."}
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

# NOUVELLE PORTE : Récupérer et enregistrer les stats d'un match
@app.get("/sync-stats/{match_id_api}")
async def sync_stats(match_id_api: int):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key or not SessionLocal: return {"erreur": "Configuration manquante"}

    # On cherche d'abord si le match est dans notre base locale
    db = SessionLocal()
    match_local = db.query(Match).filter(Match.id_source_apifootball == match_id_api).first()
    if not match_local:
        db.close()
        return {"erreur": "Ce match n'est pas dans notre base. Fais d'abord /sync-matches."}

    # On appelle l'API pour les stats
    url = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={match_id_api}"
    headers = {"x-apisports-key": api_key}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    data = response.json()

    if data.get("results", 0) == 0:
        db.close()
        return {"message": "L'API n'a pas de statistiques pour ce match."}

    try:
        # On prépare les variables
        corners_dom, corners_ext = 0, 0
        jaunes_dom, jaunes_ext = 0, 0
        rouges_dom, rouges_ext = 0, 0

        # L'API renvoie les stats de l'équipe à domicile (index 0) et extérieur (index 1)
        stats_dom = data["response"][0]["statistics"]
        stats_ext = data["response"][1]["statistics"]

        for s in stats_dom:
            if s["type"] == "Corner Kicks": corners_dom = s["value"] or 0
            if s["type"] == "Yellow Cards": jaunes_dom = s["value"] or 0
            if s["type"] == "Red Cards": rouges_dom = s["value"] or 0

        for s in stats_ext:
            if s["type"] == "Corner Kicks": corners_ext = s["value"] or 0
            if s["type"] == "Yellow Cards": jaunes_ext = s["value"] or 0
            if s["type"] == "Red Cards": rouges_ext = s["value"] or 0

        # On vérifie si on n'a pas déjà enregistré ces stats
        deja_la = db.query(MatchStat).filter(MatchStat.match_id == match_local.id).first()
        if deja_la:
            db.close()
            return {"message": "Les statistiques de ce match sont déjà enregistrées."}

        # On enregistre
        nouvelle_stat = MatchStat(
            match_id=match_local.id,
            source_api="api-football",
            corners_domicile=corners_dom,
            corners_exterieur=corners_ext,
            cartons_jaunes_domicile=jaunes_dom,
            cartons_jaunes_exterieur=jaunes_ext,
            cartons_rouges_domicile=rouges_dom,
            cartons_rouges_exterieur=rouges_ext,
            statut_validation="confirmé"
        )
        db.add(nouvelle_stat)
        db.commit()
        
        return {
            "message": "Statistiques enregistrées avec succès !",
            "corners": f"{corners_dom} - {corners_ext}",
            "cartons_jaunes": f"{jaunes_dom} - {jaunes_ext}",
            "cartons_rouges": f"{rouges_dom} - {rouges_ext}"
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()
