from fastapi import FastAPI
import os
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, Competition, Match, MatchStat
from datetime import datetime, timedelta
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

# 1. Remplir les compétitions
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

# 2. Tout synchroniser d'un coup par date
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
                    db.commit()
                    db.refresh(match_local)
                    matchs_ajoutes += 1
                
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
                            
                        await asyncio.sleep(0.5)

        return {"message": f"Terminé ! {matchs_ajoutes} matchs ajoutés, {stats_ajoutees} fiches de stats ajoutées."}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()

# 3. Le cerveau (Calcul des probabilités)
@app.get("/predict-corners/{match_id_local}")
async def predict_corners(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match:
            return {"erreur": "Match introuvable dans la base."}

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

# 4. Synchroniser une équipe par plage de dates
@app.get("/sync-equipe/{equipe_id_api}")
async def sync_equipe(equipe_id_api: int):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key or not SessionLocal: return {"erreur": "Configuration manquante"}
    
    db = SessionLocal()
    try:
        # On calcule la date d'il y a 30 jours et aujourd'hui
        aujourd_hui = datetime.now().strftime("%Y-%m-%d")
        il_y_a_30_jours = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # On demande les matchs de l'équipe sur cette période
        url = f"https://v3.football.api-sports.io/fixtures?team={equipe_id_api}&from={il_y_a_30_jours}&to={aujourd_hui}"
        headers = {"x-apisports-key": api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
        data = response.json()
        
        if data.get("results", 0) == 0: 
            return {"message": f"Aucun match trouvé pour l'équipe {equipe_id_api} entre le {il_y_a_30_jours} et le {aujourd_hui}."}

        top_leagues = [39, 140, 135, 61, 78]
        matchs_ajoutes = 0
        stats_ajoutees = 0

        for m in data["response"]:
            id_ligue = m["league"]["id"]
            if id_ligue not in top_leagues: continue
            
            if m["fixture"]["status"]["short"] != "FT": continue

            id_match_api = m["fixture"]["id"]
            competition_locale = db.query(Competition).filter(Competition.id_source == id_ligue).first()
            
            if competition_locale:
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
                    db.commit()
                    db.refresh(match_local)
                    matchs_ajoutes += 1
                
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
                        
                    await asyncio.sleep(0.5)

        return {"message": f"Succès ! {matchs_ajoutes} matchs et {stats_ajoutees} stats ajoutés pour l'équipe {equipe_id_api} (période du {il_y_a_30_jours} au {aujourd_hui})."}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()
