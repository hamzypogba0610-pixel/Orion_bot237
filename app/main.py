from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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

# --- LA NOUVELLE PAGE D'ACCUEIL (INTERFACE) ---
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ORION - Analyse Football</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #121212; color: #ffffff; padding: 20px; text-align: center; }
            h1 { color: #00e676; }
            .card { background-color: #1e1e1e; padding: 20px; border-radius: 10px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            input[type="number"] { padding: 10px; font-size: 16px; border-radius: 5px; border: 1px solid #333; background-color: #2a2a2a; color: white; width: 80%; margin-bottom: 15px; }
            button { padding: 10px 20px; font-size: 16px; border: none; border-radius: 5px; background-color: #00e676; color: #000; font-weight: bold; cursor: pointer; }
            button:hover { background-color: #00c853; }
            .result { margin-top: 20px; text-align: left; background-color: #2a2a2a; padding: 15px; border-radius: 5px; display: none; }
        </style>
    </head>
    <body>
        <h1>⚽ ORION</h1>
        <p>Le moteur d'analyse footballistique</p>
        
        <div class="card">
            <h3>Tester une prédiction</h3>
            <p>Entre l'ID local du match (ex: 1 pour Sassuolo vs Lazio)</p>
            <input type="number" id="matchId" placeholder="ID du match (ex: 1)">
            <br>
            <button onclick="analyserMatch()">Analyser le match</button>
            
            <div id="resultat" class="result"></div>
        </div>

        <script>
            async function analyserMatch() {
                const matchId = document.getElementById('matchId').value;
                if(!matchId) { alert("Entre un ID de match !"); return; }
                
                const resultDiv = document.getElementById('resultat');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = "⏳ Analyse en cours...";

                try {
                    // On appelle nos deux cerveaux
                    const resCorners = await fetch(`/predict-corners/${matchId}`);
                    const dataCorners = await resCorners.json();
                    
                    const resCartons = await fetch(`/predict-cards/${matchId}`);
                    const dataCartons = await resCartons.json();

                    if(dataCorners.erreur || dataCartons.erreur) {
                        resultDiv.innerHTML = "❌ Erreur : Match introuvable ou données manquantes.";
                        return;
                    }

                    resultDiv.innerHTML = `
                        <h4>${dataCorners.match}</h4>
                        <p><b>📊 Corners (Total attendu: ${dataCorners.analyse_corners.total_attendu})</b></p>
                        <ul>
                            <li>Plus de 8.5 corners : <b>${dataCorners.analyse_corners.probabilite_plus_de_8_5}%</b></li>
                            <li>Plus de 9.5 corners : <b>${dataCorners.analyse_corners.probabilite_plus_de_9_5}%</b></li>
                        </ul>
                        <p><b>🟨 Cartons (Total attendu: ${dataCartons.analyse_cartons.total_attendu})</b></p>
                        <ul>
                            <li>Plus de 3.5 cartons : <b>${dataCartons.analyse_cartons.probabilite_plus_de_3_5}%</b></li>
                            <li>Plus de 4.5 cartons : <b>${dataCartons.analyse_cartons.probabilite_plus_de_4_5}%</b></li>
                        </ul>
                    `;
                } catch (e) {
                    resultDiv.innerHTML = "❌ Erreur de connexion avec le serveur.";
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- TES ROUTES EXISTANTES (API) ---
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

@app.get("/predict-corners/{match_id_local}")
async def predict_corners(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

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
        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "donnees_utilisees": {"matchs_domicile": nb_matchs_dom, "matchs_exterieur": nb_matchs_ext},
            "analyse_corners": {
                "moyenne_domicile": round(moyenne_dom, 2), "moyenne_exterieur": round(moyenne_ext, 2),
                "total_attendu": round(lambda_total, 2),
                "probabilite_plus_de_9_5": round((1 - poisson.cdf(9, lambda_total)) * 100, 2),
                "probabilite_plus_de_8_5": round((1 - poisson.cdf(8, lambda_total)) * 100, 2)
            }
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/predict-cards/{match_id_local}")
async def predict_cards(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id_local).first()
        if not match: return {"erreur": "Match introuvable."}

        stats_dom = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_domicile) | (Match.equipe_exterieur == match.equipe_domicile)
        ).all()
        total_jaunes_dom, nb_matchs_dom = 0, 0
        for stat, m in stats_dom:
            if m.equipe_domicile == match.equipe_domicile and stat.cartons_jaunes_domicile is not None:
                total_jaunes_dom += stat.cartons_jaunes_domicile
                nb_matchs_dom += 1
            elif m.equipe_exterieur == match.equipe_domicile and stat.cartons_jaunes_exterieur is not None:
                total_jaunes_dom += stat.cartons_jaunes_exterieur
                nb_matchs_dom += 1
        moyenne_dom = (total_jaunes_dom / nb_matchs_dom) if nb_matchs_dom > 0 else 2.0

        stats_ext = db.query(MatchStat, Match).join(Match, MatchStat.match_id == Match.id).filter(
            (Match.equipe_domicile == match.equipe_exterieur) | (Match.equipe_exterieur == match.equipe_exterieur)
        ).all()
        total_jaunes_ext, nb_matchs_ext = 0, 0
        for stat, m in stats_ext:
            if m.equipe_domicile == match.equipe_exterieur and stat.cartons_jaunes_domicile is not None:
                total_jaunes_ext += stat.cartons_jaunes_domicile
                nb_matchs_ext += 1
            elif m.equipe_exterieur == match.equipe_exterieur and stat.cartons_jaunes_exterieur is not None:
                total_jaunes_ext += stat.cartons_jaunes_exterieur
                nb_matchs_ext += 1
        moyenne_ext = (total_jaunes_ext / nb_matchs_ext) if nb_matchs_ext > 0 else 2.0

        lambda_total = moyenne_dom + moyenne_ext
        return {
            "match": f"{match.equipe_domicile} vs {match.equipe_exterieur}",
            "donnees_utilisees": {"matchs_domicile": nb_matchs_dom, "matchs_exterieur": nb_matchs_ext},
            "analyse_cartons": {
                "moyenne_jaunes_domicile": round(moyenne_dom, 2),
                "moyenne_jaunes_exterieur": round(moyenne_ext, 2),
                "total_attendu": round(lambda_total, 2),
                "probabilite_plus_de_3_5": round((1 - poisson.cdf(3, lambda_total)) * 100, 2),
                "probabilite_plus_de_4_5": round((1 - poisson.cdf(4, lambda_total)) * 100, 2),
                "probabilite_plus_de_5_5": round((1 - poisson.cdf(5, lambda_total)) * 100, 2)
            }
        }
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        db.close()

@app.get("/import-csv/")
async def import_csv(url: str):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        serie_a = db.query(Competition).filter(Competition.nom == "Serie A").first()
        if not serie_a: return {"erreur": "Serie A non trouvée."}
        competition_id = serie_a.id 

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
        
        if response.status_code != 200: return {"erreur": f"Erreur HTTP : {response.status_code}"}
            
        csv_file = io.StringIO(response.text)
        reader = csv.DictReader(csv_file)
        
        matchs_ajoutes, stats_ajoutees = 0, 0

        for row in reader:
            if not row.get('HomeTeam') or not row.get('AwayTeam') or not row.get('Date'): continue
                
            fake_id_str = f"{row['HomeTeam']}{row['AwayTeam']}{row['Date']}"
            fake_id = int(hashlib.md5(fake_id_str.encode()).hexdigest()[:8], 16) % 2000000000
            
            if db.query(Match).filter(Match.id_source_apifootball == fake_id).first(): continue

            try:
                date_obj = datetime.strptime(row['Date'], "%d/%m/%Y")
            except:
                try:
                    date_obj = datetime.strptime(row['Date'], "%d/%m/%y")
                except:
                    date_obj = datetime.now()

            nouveau_match = Match(
                competition_id=competition_id, date_heure=date_obj,
                equipe_domicile=row['HomeTeam'], equipe_exterieur=row['AwayTeam'], statut="FT",
                id_source_apifootball=fake_id,
                score_domicile=int(row['FTHG']) if row.get('FTHG') and row['FTHG'].isdigit() else None,
                score_exterieur=int(row['FTAG']) if row.get('FTAG') and row['FTAG'].isdigit() else None
            )
            db.add(nouveau_match)
            db.commit()
            db.refresh(nouveau_match)
            matchs_ajoutes += 1

            if row.get('HC') and row.get('AC'):
                nouvelle_stat = MatchStat(
                    match_id=nouveau_match.id, source_api="csv-football-data",
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
        return {"message": f"Importation réussie ! {matchs_ajoutes} matchs et {stats_ajoutees} stats ajoutés."}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()
