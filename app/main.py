from fastapi import FastAPI
import os
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, Competition, Match, MatchStat
from datetime import datetime
from scipy.stats import poisson

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

# (Toutes tes routes précédentes : init-competitions, sync-matches, sync-stats sont gardées ici...)
# ... [Pour faire court, je ne les remets pas toutes ici, mais tu dois garder le fichier complet que tu as actuellement]

# NOUVELLE PORTE : Le Cerveau (Loi de Poisson)
@app.get("/predict-corners/{match_id_local}")
async def predict_corners(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    
    db = SessionLocal()
    try:
        # 1. On récupère le match dans notre base
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match:
            return {"erreur": "Match introuvable dans la base."}
            
        # 2. On définit les moyennes (Pour l'instant, on simule avec des chiffres fixes pour tester la formule)
        # Plus tard, on les calculera avec l'historique réel des équipes.
        moyenne_corners_domicile = 5.5
        moyenne_corners_exterieur = 4.0
        
        # La moyenne totale attendue (Lambda)
        lambda_total = moyenne_corners_domicile + moyenne_corners_exterieur
        
        # 3. On calcule la probabilité d'avoir PLUS de 9.5 corners (donc 10 ou plus)
        # En Poisson, P(X > 9.5) = 1 - P(X <= 9)
        proba_plus_10 = 1 - poisson.cdf(9, lambda_total)
        
        # On calcule aussi pour une autre ligne (ex: plus de 8.5)
        proba_plus_9 = 1 - poisson.cdf(8, lambda_total)

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "analyse_corners": {
                "moyenne_attendue_domicile": moyenne_corners_domicile,
                "moyenne_attendue_exterieur": moyenne_corners_exterieur,
                "total_attendu": lambda_total,
                "probabilite_plus_de_9_5": round(proba_plus_10 * 100, 2), # En pourcentage
                "probabilite_plus_de_8_5": round(proba_plus_9 * 100, 2)
            }
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()
