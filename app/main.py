from fastapi import FastAPI
import os
import httpx
from sqlalchemy import create_engine, func
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

# (Tes autres routes /sync-matches et /sync-stats sont à garder ici)
# ... [Je les omets pour faire court, mais garde-les dans ton fichier]

# NOUVELLE VERSION DU CERVEAU : Calcul des vraies moyennes
@app.get("/predict-corners/{match_id_local}")
async def predict_corners(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    
    db = SessionLocal()
    try:
        # 1. On récupère le match à analyser
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match:
            return {"erreur": "Match introuvable dans la base."}

        # 2. On calcule la moyenne de corners de l'équipe à domicile
        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_domicile) | (Match.equipe_exterieur == match.equipe_domicile)
        ).all()
        
        total_corners_dom = 0
        nb_matchs_dom = 0
        for stat, m in stats_dom:
            if m.equipe_domicile == match.equipe_domicile and stat.corners_domicile is not None:
                total_corners_dom += stat.corners_domicile
                nb_matchs_dom += 1
            elif m.equipe_exterieur == match.equipe_domicile and stat.corners_exterieur is not None:
                total_corners_dom += stat.corners_exterieur
                nb_matchs_dom += 1

        moyenne_dom = (total_corners_dom / nb_matchs_dom) if nb_matchs_dom > 0 else 5.5

        # 3. On calcule la moyenne de corners de l'équipe à l'extérieur
        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_exterieur) | (Match.equipe_exterieur == match.equipe_exterieur)
        ).all()
        
        total_corners_ext = 0
        nb_matchs_ext = 0
        for stat, m in stats_ext:
            if m.equipe_domicile == match.equipe_exterieur and stat.corners_domicile is not None:
                total_corners_ext += stat.corners_domicile
                nb_matchs_ext += 1
            elif m.equipe_exterieur == match.equipe_exterieur and stat.corners_exterieur is not None:
                total_corners_ext += stat.corners_exterieur
                nb_matchs_ext += 1

        moyenne_ext = (total_corners_ext / nb_matchs_ext) if nb_matchs_ext > 0 else 4.0

        # 4. On applique la Loi de Poisson
        lambda_total = moyenne_dom + moyenne_ext
        proba_plus_10 = 1 - poisson.cdf(9, lambda_total)
        proba_plus_9 = 1 - poisson.cdf(8, lambda_total)

        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "donnees_utilisees": {
                "matchs_domicile_analyses": nb_matchs_dom,
                "matchs_exterieur_analyses": nb_matchs_ext
            },
            "analyse_corners": {
                "moyenne_reelle_domicile": round(moyenne_dom, 2),
                "moyenne_reelle_exterieur": round(moyenne_ext, 2),
                "total_attendu": round(lambda_total, 2),
                "probabilite_plus_de_9_5": round(proba_plus_10 * 100, 2),
                "probabilite_plus_de_8_5": round(proba_plus_9 * 100, 2)
            }
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()
