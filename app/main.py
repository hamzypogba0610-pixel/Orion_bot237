from fastapi import FastAPI
import os
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, Competition, Match, MatchStat
from datetime import datetime
from scipy.stats import poisson
import asyncio

app = FastAPI(title="ORION", version="0.1.0")

# --- CONNEXION À LA BASE DE DONNÉES ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True, 
    pool_recycle=300
) if DATABASE_URL else None

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None

if engine:
    Base.metadata.create_all(bind=engine)

@app.get("/")
async def home():
    return {"message": "ORION est en ligne", "version": "0.1.0", "statut": "vivant"}

# NOUVELLE PORTE : Tout synchroniser d'un coup
@app.get("/sync-tout/{date_recherche}")
async def sync_tout(date_recherche: str):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key or not SessionLocal: return {"erreur": "Configuration manquante"}
    
    db = SessionLocal()
    try:
        url = f"https://v3.football.api-sports.io/fixtures?date={date_recherche}"
        headers = {"x-apisports-key": api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
        data = response.json()
        
        if data.get("results", 0) == 0: return {"message": "Aucun match ce jour-là."}

        top_leagues = [39, 140, 135, 61, 78]
        matchs_ajoutes = 0
        stats_ajoutees = 0

        for m in data["response"]:
            id_ligue = m["league"]["id"]
            if id_ligue not in top_leagues: continue

            id_match_api = m["fixture"]["id"]
            competition_locale = db.query(Competition).filter(Competition.id_source == id_ligue).first()
            
            if competition_locale:
                # 1. Enregistrer le match s'il n'existe pas
                match_local = db.query(Match).filter(Match.id_source_apifootball == id_match_api).first()
                if not match_local:
                    date_str = m["fixture"]["date"].replace("Z", "+00:00")
                    match_local = Match(
                        competition_id=competition_locale.id,
                        date_heure=datetime.fromisoformat(date_str),
                        equipe_domicile=m["teams"]["home"]["name"],
                        equipe_exterieur=m["teams"]["away"]["name"],
                        statut=m["fixture"]["status"]["short"],
                        id_source_apifootball=id_match_api,
                        score_domicile=m["goals"]["home"],
                        score_exterieur=m["goals"]["away"]
                    )
                    db.add(match_local)
                    db.commit() # On sauvegarde pour avoir l'ID local
                    db.refresh(match_local)
                    matchs_ajoutes += 1
                
                # 2. Si le match est terminé, on cherche les stats
                if m["fixture"]["status"]["short"] == "FT":
                    deja_stats = db.query(MatchStat).filter(MatchStat.match_id == match_local.id).first()
                    if not deja_stats:
                        url_stats = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={id_match_api}"
                        async with httpx.AsyncClient() as client:
                            resp_stats = await client.get(url_stats, headers=headers)
                        data_stats = resp_stats.json()
                        
                        if data_stats.get("results", 0) > 0:
                            corners_dom, corners_ext = 0, 0
                            jaunes_dom, jaunes_ext = 0, 0
                            rouges_dom, rouges_ext = 0, 0
                            
                            stats_dom = data_stats["response"][0]["statistics"]
                            stats_ext = data_stats["response"][1]["statistics"]
                            
                            for s in stats_dom:
                                if s["type"] == "Corner Kicks": corners_dom = s["value"] or 0
                                if s["type"] == "Yellow Cards": jaunes_dom = s["value"] or 0
                                if s["type"] == "Red Cards": rouges_dom = s["value"] or 0
                            for s in stats_ext:
                                if s["type"] == "Corner Kicks": corners_ext = s["value"] or 0
                                if s["type"] == "Yellow Cards": jaunes_ext = s["value"] or 0
                                if s["type"] == "Red Cards": rouges_ext = s["value"] or 0

                            db.add(MatchStat(
                                match_id=match_local.id,
                                source_api="api-football",
                                corners_domicile=corners_dom,
                                corners_exterieur=corners_ext,
                                cartons_jaunes_domicile=jaunes_dom,
                                cartons_jaunes_exterieur=jaunes_ext,
                                cartons_rouges_domicile=rouges_dom,
                                cartons_rouges_exterieur=rouges_ext,
                                statut_validation="confirmé"
                            ))
                            db.commit()
                            stats_ajoutees += 1
                            
                        await asyncio.sleep(0.5) # Petite pause pour ne pas surcharger l'API

        return {"message": f"Terminé ! {matchs_ajoutes} matchs ajoutés, {stats_ajoutees} fiches de stats ajoutées."}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()

# (Garde tes routes /predict-corners, /init-competitions, etc. ici en dessous)
