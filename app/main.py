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

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None

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
                --bg-dark: #0f172a; --bg-card: #1e293b; --text-main: #f8fafc; --text-muted: #94a3b8;
                --accent: #10b981; --accent-hover: #059669; --border: #334155; --danger: #ef4444;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
            body { background-color: var(--bg-dark); color: var(--text-main); padding: 20px; display: flex; justify-content: center; min-height: 100vh; }
            .container { width: 100%; max-width: 500px; }
            
            .header { text-align: center; margin-bottom: 25px; margin-top: 10px; }
            .header h1 { font-size: 28px; font-weight: 800; color: var(--accent); display: flex; align-items: center; justify-content: center; gap: 10px; }
            .header p { color: var(--text-muted); font-size: 14px; margin-top: 5px; }

            .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 20px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); margin-bottom: 20px; }
            .input-group { display: flex; flex-direction: column; gap: 15px; }
            label { font-size: 13px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }
            input[type="number"] { width: 100%; padding: 14px; border-radius: 10px; border: 1px solid var(--border); background: var(--bg-dark); color: white; font-size: 16px; }
            input[type="number"]:focus { outline: none; border-color: var(--accent); }
            button { width: 100%; padding: 14px; border: none; border-radius: 10px; background: var(--accent); color: #000; font-size: 16px; font-weight: 800; cursor: pointer; transition: 0.2s; }
            button:hover { background: var(--accent-hover); }

            #resultats { display: none; animation: fadeIn 0.5s ease-in-out; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            
            .match-title { text-align: center; font-size: 18px; font-weight: 800; margin-bottom: 15px; padding: 15px; background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; border: 1px solid var(--border); }
            .market-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 20px; margin-bottom: 15px; }
            .market-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
            .market-title { font-size: 16px; font-weight: 600; }
            .market-total { font-size: 13px; color: var(--accent); font-weight: 800; background: rgba(16, 185, 129, 0.1); padding: 4px 10px; border-radius: 20px; }
            
            .prob-row { margin-bottom: 15px; background: var(--bg-dark); padding: 12px; border-radius: 8px; border: 1px solid var(--border); }
            .prob-labels { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; }
            .prob-label { color: var(--text-muted); }
            .prob-value { font-weight: 800; }
            .bar-bg { width: 100%; height: 6px; background: #000; border-radius: 4px; overflow: hidden; margin-bottom: 10px; }
            .bar-fill { height: 100%; border-radius: 4px; transition: width 1s ease-out; }
            
            .cote-input { width: 80px !important; padding: 6px !important; font-size: 14px !important; text-align: center; border-radius: 6px !important; }
            .value-badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 800; margin-top: 5px; }
            .value-yes { background-color: rgba(16, 185, 129, 0.2); color: var(--accent); border: 1px solid var(--accent); }
            .value-no { background-color: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }
            .value-empty { background-color: var(--bg-card); color: var(--text-muted); border: 1px solid var(--border); }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚽ ORION</h1>
                <p>Football Analytics Engine</p>
            </div>

            <div class="card">
                <div class="input-group">
                    <div>
                        <label>ID du Match</label>
                        <input type="number" id="matchId" placeholder="Ex: 1">
                    </div>
                    <button onclick="analyserMatch()">🧠 Analyser le match</button>
                </div>
            </div>

            <div id="resultats">
                <div class="match-title" id="matchTitle"></div>
                <div class="market-card">
                    <div class="market-header">
                        <span class="market-title">📐 Corners</span>
                        <span class="market-total" id="totalCorners"></span>
                    </div>
                    <div id="cornersProbs"></div>
                </div>
                <div class="market-card">
                    <div class="market-header">
                        <span class="market-title">🟨 Cartons</span>
                        <span class="market-total" id="totalCartons"></span>
                    </div>
                    <div id="cartonsProbs"></div>
                </div>
            </div>
        </div>

        <script>
            async function analyserMatch() {
                const matchId = document.getElementById('matchId').value;
                if(!matchId) { alert("Veuillez entrer un ID."); return; }

                document.getElementById('resultats').style.display = 'none';

                try {
                    const resC = await fetch(`/predict-corners/${matchId}`);
                    const dataC = await resC.json();
                    const resK = await fetch(`/predict-cards/${matchId}`);
                    const dataK = await resK.json();

                    if(dataC.erreur || dataK.erreur) { alert("Match introuvable."); return; }

                    document.getElementById('matchTitle').innerText = dataC.match;
                    document.getElementById('totalCorners').innerText = `Moy: ${dataC.analyse_corners.total_attendu}`;
                    document.getElementById('totalCartons').innerText = `Moy: ${dataK.analyse_cartons.total_attendu}`;

                    document.getElementById('cornersProbs').innerHTML = genererBarres(dataC.analyse_corners, ['probabilite_plus_de_8_5', 'probabilite_plus_de_9_5'], ['+8.5 Corners', '+9.5 Corners']);
                    document.getElementById('cartonsProbs').innerHTML = genererBarres(dataK.analyse_cartons, ['probabilite_plus_de_3_5', 'probabilite_plus_de_4_5'], ['+3.5 Cartons', '+4.5 Cartons']);

                    document.getElementById('resultats').style.display = 'block';
                } catch (e) {
                    alert("Erreur de connexion.");
                }
            }

            function genererBarres(data, cles, labels) {
                let html = '';
                cles.forEach((cle, index) => {
                    let proba = data[cle];
                    let couleurBarre = proba > 60 ? '#10b981' : (proba > 40 ? '#f59e0b' : '#ef4444');
                    let idInput = `cote_${cle}`; // ID unique pour chaque case
                    
                    html += `
                        <div class="prob-row">
                            <div class="prob-labels">
                                <span class="prob-label">${labels[index]}</span>
                                <span class="prob-value" style="color: ${couleurBarre}">${proba}%</span>
                            </div>
                            <div class="bar-bg"><div class="bar-fill" style="width: ${proba}%; background-color: ${couleurBarre};"></div></div>
                            
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                                <div>
                                    <span style="font-size: 12px; color: var(--text-muted);">Cote: </span>
                                    <input type="number" step="0.01" class="cote-input" id="${idInput}" placeholder="Ex: 1.85" oninput="calculerEV('${idInput}', ${proba})">
                                </div>
                                <div id="badge_${cle}">
                                    <span class="value-badge value-empty">En attente de cote</span>
                                </div>
                            </div>
                        </div>
                    `;
                });
                return html;
            }

            function calculerEV(inputId, proba) {
                let cote = parseFloat(document.getElementById(inputId).value);
                let badgeDiv = document.getElementById(`badge_${inputId.replace('cote_', '')}`);
                
                if (!cote || cote < 1) {
                    badgeDiv.innerHTML = `<span class="value-badge value-empty">En attente de cote</span>`;
                    return;
                }

                let probaDecimale = proba / 100;
                let ev = (probaDecimale * cote) - 1;

                if (ev > 0) {
                    badgeDiv.innerHTML = `<span class="value-badge value-yes">✅ VALUE BET (+${(ev*100).toFixed(1)}% EV)</span>`;
                } else {
                    badgeDiv.innerHTML = `<span class="value-badge value-no">❌ Pas de Value (${(ev*100).toFixed(1)}% EV)</span>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
