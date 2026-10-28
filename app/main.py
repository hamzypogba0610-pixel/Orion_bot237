from fastapi import FastAPI
import os
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, Competition

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

# NOUVELLE PORTE : Ajouter les compétitions dans la base
@app.get("/init-competitions")
async def init_competitions():
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    
    db = SessionLocal()
    try:
        # La liste de nos grands championnats
        les_ligues = [
            {"id_api": 39, "nom": "Premier League", "pays": "Angleterre"},
            {"id_api": 140, "nom": "La Liga", "pays": "Espagne"},
            {"id_api": 135, "nom": "Serie A", "pays": "Italie"},
            {"id_api": 78, "nom": "Bundesliga", "pays": "Allemagne"},
            {"id_api": 61, "nom": "Ligue 1", "pays": "France"}
        ]
        
        ajoutees = 0
        for ligue in les_ligues:
            # On vérifie qu'on ne l'a pas déjà ajoutée
            existe = db.query(Competition).filter(Competition.id_source == ligue["id_api"]).first()
            if not existe:
                nouvelle = Competition(nom=ligue["nom"], pays=ligue["pays"], id_source=ligue["id_api"])
                db.add(nouvelle)
                ajoutees += 1
                
        db.commit()
        return {"message": f"Succès ! {ajoutees} compétitions ajoutées à la base de données."}
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()
