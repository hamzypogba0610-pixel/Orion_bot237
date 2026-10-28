from fastapi import FastAPI
import os
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, Competition, Match, MatchStat
from datetime import datetime
from scipy.stats import poisson
import asyncio
import csv
import io
import hashlib

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

# 1. Remplir les compétitions (si ce n'est pas déjà fait)
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

# 2. Le Cerveau (Calcul des probabilités)
@app.get("/predict-corners/{match_id_local}")
async def predict_corners(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match:
            return {"erreur": "Match introuvable dans la base."}

        # Calcul des moyennes pour l'équipe à domicile
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

        # Calcul des moyennes pour l'équipe à l'extérieur
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

# 3. LA NOUVELLE PORTE MAGIQUE : Importer un fichier CSV
@app.get("/import-csv/")
async def import_csv(url: str):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    
    db = SessionLocal()
    try:
        # 1. Télécharger le fichier CSV
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        
        if response.status_code != 200:
            return {"erreur": "Impossible de télécharger le fichier CSV."}
            
        # 2. Lire le fichier ligne par ligne
        csv_file = io.StringIO(response.text)
        reader = csv.DictReader(csv_file)
        
        matchs_ajoutes = 0
        stats_ajoutees = 0
        competition_id = 135 # On force sur la Serie A (Italie) pour ce test

        for row in reader:
            # On vérifie que les données essentielles existent
            if not row.get('HomeTeam') or not row.get('AwayTeam') or not row.get('Date'):
                continue
                
            # On crée un faux ID unique pour la base de données (basé sur les noms et la date)
            fake_id_str = f"{row['HomeTeam']}{row['AwayTeam']}{row['Date']}"
            fake_id = int(hashlib.md5(fake_id_str.encode()).hexdigest()[:8], 16)
            
            # On vérifie si le match n'est pas déjà dans la base
            match_existant = db.query(Match).filter(Match.id_source_apifootball == fake_id).first()
            if match_existant:
                continue

            # On formate la date (le CSV donne JJ/MM/AAAA)
            try:
                date_obj = datetime.strptime(row['Date'], "%d/%m/%Y")
            except:
                try:
                    date_obj = datetime.strptime(row['Date'], "%d/%m/%y")
                except:
                    date_obj = datetime.now()

            # On crée le match
            nouveau_match = Match(
                competition_id=competition_id,
                date_heure=date_obj,
                equipe_domicile=row['HomeTeam'],
                equipe_exterieur=row['AwayTeam'],
                statut="FT",
                id_source_apifootball=fake_id,
                score_domicile=int(row['FTHG']) if row.get('FTHG') and row['FTHG'].isdigit() else None,
                score_exterieur=int(row['FTAG']) if row.get('FTAG') and row['FTAG'].isdigit() else None
            )
            db.add(nouveau_match)
            db.commit()
            db.refresh(nouveau_match)
            matchs_ajoutes += 1

            # On crée les stats (Corners et Cartons)
            if row.get('HC') and row.get('AC'):
                nouvelle_stat = MatchStat(
                    match_id=nouveau_match.id,
                    source_api="csv-football-data",
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
        return {
            "message": f"Importation réussie ! {matchs_ajoutes} matchs et {stats_ajoutees} fiches de stats ajoutés à la base.",
            "competition": "Serie A"
        }
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()
