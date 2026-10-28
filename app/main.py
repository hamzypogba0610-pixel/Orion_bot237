from fastapi import FastAPI
import os
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db_models import Base

app = FastAPI(title="ORION", version="0.1.0")

# --- CONNEXION À LA BASE DE DONNÉES ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None

# Création des 6 vraies tables dans Neon !
if engine:
    Base.metadata.create_all(bind=engine)

@app.get("/")
async def home():
    return {"message": "ORION est en ligne", "version": "0.1.0", "statut": "vivant"}

# (Les routes test-matchs, test-stats, etc., sont gardées ici pour plus tard)
@app.get("/test-top-matchs/{date_recherche}")
async def test_top_matchs(date_recherche: str):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key: return {"erreur": "Clé introuvable"}
    url = f"https://v3.football.api-sports.io/fixtures?date={date_recherche}"
    headers = {"x-apisports-key": api_key}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    data = response.json()
    if data.get("results", 0) == 0: return {"message": "Aucun match."}
    top_leagues = [39, 140, 135, 61, 78]
    top_matchs = []
    for m in data["response"]:
        if m["league"]["id"] in top_leagues:
            top_matchs.append({
                "id_match": m["fixture"]["id"],
                "competition": m["league"]["name"],
                "equipe_domicile": m["teams"]["home"]["name"],
                "equipe_exterieur": m["teams"]["away"]["name"],
                "statut": m["fixture"]["status"]["short"],
                "score": f"{m['goals']['home']} - {m['goals']['away']}"
            })
    if not top_matchs: return {"message": "Aucun match des grands championnats trouvé."}
    return top_matchs[:10]

@app.get("/test-db")
async def test_db():
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    return {"message": "Regarde dans ton tableau de bord Neon ! Les 6 tables ont été créées."}
