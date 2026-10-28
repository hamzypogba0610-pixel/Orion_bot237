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

# --- LA NOUVELLE INTERFACE PRO (DASHBOARD) ---
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>ORION | Football Analytics</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-dark: #0f172a;
                --bg-card: #1e293b;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --accent: #10b981;
                --accent-hover: #059669;
                --border: #334155;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
            body { background-color: var(--bg-dark); color: var(--text-main); padding: 20px; display: flex; justify-content: center; min-height: 100vh; }
            .container { width: 100%; max-width: 500px; }
            
            /* Header */
            .header { text-align: center; margin-bottom: 30px; margin-top: 20px; }
            .header h1 { font-size: 28px; font-weight: 800; letter-spacing: 1px; color: var(--accent); display: flex; align-items: center; justify-content: center; gap: 10px; }
            .header p { color: var(--text-muted); font-size: 14px; margin-top: 5px; }

            /* Input Card */
            .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 20px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); margin-bottom: 20px; }
            .input-group { display: flex; flex-direction: column; gap: 12px; }
            label { font-size: 13px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
            input { width: 100%; padding: 14px; border-radius: 10px; border: 1px solid var(--border); background: var(--bg-dark); color: white; font-size: 16px; transition: 0.3s; }
            input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2); }
            button { width: 100%; padding: 14px; border: none; border-radius: 10px; background: var(--accent); color: #000; font-size: 16px; font-weight: 800; cursor: pointer; transition: 0.2s; display: flex; justify-content: center; align-items: center; gap: 8px; }
            button:hover { background: var(--accent-hover); transform: translateY(-2px); }
            button:active { transform: translateY(0); }

            /* Results Section */
            #resultats { display: none; animation: fadeIn 0.5s ease-in-out; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            
            .match-title { text-align: center; font-size: 20px; font-weight: 800; margin-bottom: 20px; padding: 15px; background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; border: 1px solid var(--border); }
            
            .grid { display: grid; grid-template-columns: 1fr; gap: 15px; }
            .market-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 20px; }
            .market-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
            .market-title { font-size: 18px; font-weight: 600; }
            .market-total { font-size: 14px; color: var(--accent); font-weight: 800; background: rgba(16, 185, 129, 0.1); padding: 4px 10px; border-radius: 20px; }
            
            /* Probability Bars */
            .prob-row { margin-bottom: 15px; }
            .prob-labels { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; }
            .prob-label { color: var(--text-muted); }
            .prob-value { font-weight: 800; }
            .bar-bg { width: 100%; height: 8px; background: var(--bg-dark); border-radius: 4px; overflow: hidden; }
            .bar-fill { height: 100%; border-radius: 4px; transition: width 1s ease-out; }
            
            .loading { display: none; text-align: center; margin-top: 20px; }
            .spinner { width: 30px; height: 30px; border: 3px solid rgba(255,255,255,0.1); border-top-color: var(--accent); border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 10px auto; }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="header">
                <h1><span style="font-size: 32px;">⚽</span> ORION</h1>
                <p>Football Analytics Engine</p>
            </div>

            <!-- Input Card -->
            <div class="card">
                <div class="input-group">
                    <label>ID du Match (Base locale)</label>
                    <input type="number" id="matchId" placeholder="Ex: 1 (Sassuolo vs Lazio)">
                    <button onclick="analyserMatch()" id="btnAnalyser">
                        🧠 Analyser le match
                    </button>
                </div>
            </div>

            <!-- Loading -->
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p style="color: var(--text-muted); font-size: 14px;">Calcul des probabilités...</p>
            </div>

            <!-- Results -->
            <div id="resultats">
                <div class="match-title" id="matchTitle"></div>
                
                <div class="grid">
                    <!-- Carte Corners -->
                    <div class="market-card">
                        <div class="market-header">
                            <span class="market-title">📐 Corners</span>
                            <span class="market-total" id="totalCorners"></span>
                        </div>
                        <div id="cornersProbs"></div>
                    </div>

                    <!-- Carte Cartons -->
                    <div class="market-card">
                        <div class="market-header">
                            <span class="market-title">🟨 Cartons</span>
                            <span class="market-total" id="totalCartons"></span>
                        </div>
                        <div id="cartonsProbs"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function analyserMatch() {
                const matchId = document.getElementById('matchId').value;
                if(!matchId) { alert("Veuillez entrer un ID de match."); return; }
                
                // UI updates
                document.getElementById('btnAnalyser').style.display = 'none';
                document.getElementById('loading').style.display = 'block';
                document.getElementById('resultats').style.display = 'none';

                try {
                    const resCorners = await fetch(`/predict-corners/${matchId}`);
                    const dataCorners = await resCorners.json();
                    
                    const resCartons = await fetch(`/predict-cards/${matchId}`);
                    const dataCartons = await resCartons.json();

                    if(dataCorners.erreur || dataCartons.erreur) {
                        alert("Erreur : Match introuvable ou données manquantes.");
                        return resetUI();
                    }

                    // Remplissage des données
                    document.getElementById('matchTitle').innerText = dataCorners.match;
                    document.getElementById('totalCorners').innerText = `Moyenne: ${dataCorners.analyse_corners.total_attendu}`;
                    document.getElementById('totalCartons').innerText = `Moyenne: ${dataCartons.analyse_cartons.total_attendu}`;

                    // Génération des barres de probabilité
                    document.getElementById('cornersProbs').innerHTML = genererBarres(dataCorners.analyse_corners, ['probabilite_plus_de_8_5', 'probabilite_plus_de_9_5'], ['+8.5 Corners', '+9.5 Corners']);
                    document.getElementById('cartonsProbs').innerHTML = genererBarres(dataCartons.analyse_cartons, ['probabilite_plus_de_3_5', 'probabilite_plus_de_4_5'], ['+3.5 Cartons', '+4.5 Cartons']);

                    document.getElementById('resultats').style.display = 'block';
                } catch (e) {
                    alert("Erreur de connexion avec le serveur.");
                } finally {
                    resetUI();
                }
            }

            function genererBarres(data, cles, labels) {
                let html = '';
                cles.forEach((cle, index) => {
                    let proba = data[cle];
                    // Couleur dynamique selon la probabilité
                    let couleur = proba > 60 ? '#10b981' : (proba > 40 ? '#f59e0b' : '#ef4444');
                    
                    html += `
                        <div class="prob-row">
                            <div class="prob-labels">
                                <span class="prob-label">${labels[index]}</span>
                                <span class="prob-value" style="color: ${couleur}">${proba}%</span>
                            </div>
                            <div class="bar-bg">
                                <div class="bar-fill" style="width: ${proba}%; background-color: ${couleur};"></div>
                            </div>
                        </div>
                    `;
                });
                return html;
            }

            function resetUI() {
                document.getElementById('btnAnalyser').style.display = 'flex';
                document.getElementById('loading').style.display = 'none';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- ROUTES API (INCHANGÉES) ---
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
