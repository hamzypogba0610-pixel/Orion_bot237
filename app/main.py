from fastapi import FastAPI
import os
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, Competition, Match
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

# NOUVELLE PORTE : Récupérer et enregistrer les matchs
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
            id_api_ligue = m["league"]["id"]
            
            # On cherche si ce championnat est dans notre base (les 5 grands)
            competition_locale = db.query(Competition).filter(Competition.id_source == id_api_ligue).first()
            
            if competition_locale:
                id_match_api = m["fixture"]["id"]
                
                # On vérifie si le match n'est pas déjà enregistré
                existe = db.query(Match).filter(Match.id_source_apifootball == id_match_api).first()
                
                if not existe:
                    # On prépare la date (l'API donne une date avec un fuseau horaire, on la nettoie)
                    date_str = m["fixture"]["date"].replace("Z", "+00:00")
                    date_formattee = datetime.fromisoformat(date_str)
                    
                    nouveau_match = Match(
                        competition_id=competition_locale.id,
                        date_heure=date_formattee,
                        equipe_domicile=m["teams"]["home"]["name"],
                        equipe_exterieur=m["teams"]["away"]["name"],
                        statut=m["fixture"]["status"]["short"],
                        id_source_apifootball=id_match_api,
                        score_domicile=m["goals"]["home"],
                        score_exterieur=m["goals"]["away"]
                    )
                    db.add(nouveau_match)
                    ajoutes += 1
                    
        db.commit()
        return {"message": f"Succès ! {ajoutes} matchs ont été ajoutés à la base pour le {date_recherche}."}
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()
