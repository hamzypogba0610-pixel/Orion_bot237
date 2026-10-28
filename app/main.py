from fastapi import FastAPI
import os
import httpx
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

app = FastAPI(title="ORION", version="0.1.0")

# --- CONFIGURATION DE LA BASE DE DONNÉES ---
DATABASE_URL = os.getenv("DATABASE_URL")
# Neon donne parfois un lien qui commence par postgres://, SQLAlchemy préfère postgresql://
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()

# On crée un petit tiroir de test
class TestDB(Base):
    __tablename__ = "test_db"
    id = Column(Integer, primary_key=True, index=True)
    message = Column(String, index=True)

# On demande à ORION de créer ce tiroir au démarrage
if engine:
    Base.metadata.create_all(bind=engine)

# --- TES ROUTES EXISTANTES ---
@app.get("/")
async def home():
    return {"message": "ORION est en ligne", "version": "0.1.0", "statut": "vivant"}

@app.get("/test-matchs/{date_recherche}")
async def test_matchs(date_recherche: str):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key: return {"erreur": "Clé introuvable"}
    url = f"https://v3.football.api-sports.io/fixtures?date={date_recherche}"
    headers = {"x-apisports-key": api_key}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    data = response.json()
    if data.get("results", 0) == 0: return {"message": "Aucun match."}
    
    matchs = []
    for m in data["response"][:5]:
        matchs.append({
            "id_match": m["fixture"]["id"],
            "competition": m["league"]["name"],
            "equipe_domicile": m["teams"]["home"]["name"],
            "equipe_exterieur": m["teams"]["away"]["name"],
            "score_final": f"{m['goals']['home']} - {m['goals']['away']}"
        })
    return matchs

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

@app.get("/test-stats/{match_id}")
async def test_stats(match_id: int):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key: return {"erreur": "Clé introuvable"}
    url = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={match_id}"
    headers = {"x-apisports-key": api_key}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    data = response.json()
    if data.get("results", 0) == 0: return {"message": "Aucune statistique pour ce match."}
    return data["response"]

# --- NOUVELLE ROUTE : TEST DE LA BASE DE DONNÉES ---
@app.get("/test-db")
async def test_db():
    if not SessionLocal:
        return {"erreur": "La base de données n'est pas connectée. Vérifie la clé DATABASE_URL sur Render."}
    
    db = SessionLocal()
    try:
        # On écrit un message dans le tiroir
        nouveau_message = TestDB(message="Connexion réussie ! ORION est prêt.")
        db.add(nouveau_message)
        db.commit()
        db.refresh(nouveau_message)
        
        # On lit ce qu'on vient d'écrire
        return {
            "statut": "succès",
            "id_dans_la_base": nouveau_message.id,
            "message_enregistre": nouveau_message.message
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()
