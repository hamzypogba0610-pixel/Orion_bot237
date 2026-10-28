from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, Competition, Match, MatchStat, Prediction
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
            input[type="number"], select { width: 100%; padding: 14px; border-radius: 10px; border: 1px solid var(--border); background: var(--bg-dark); color: white; font-size: 16px; appearance: none; -webkit-appearance: none; }
            input[type="number"]:focus, select:focus { outline: none; border-color: var(--accent); }
            button { width: 100%; padding: 14px; border: none; border-radius: 10px; background: var(--accent); color: #000; font-size: 16px; font-weight: 800; cursor: pointer; transition: 0.2s; }
            button:hover { background: var(--accent-hover); }
            .btn-save { background-color: #3b82f6; color: white; margin-top: 15px; }
            .btn-save:hover { background-color: #2563eb; }

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
                        <label>Sélectionner un match</label>
                        <select id="matchId">
                            <option value="">Chargement des matchs...</option>
                        </select>
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
                <button class="btn-save" onclick="sauvegarderPrediction()">💾 Sauvegarder cette analyse</button>
            </div>
        </div>

        <script>
            let currentMatchId = null;
            let currentDataC = null;
            let currentDataK = null;

            async function chargerMatchs() {
                try {
                    const response = await fetch('/api/matchs');
                    const matchs = await response.json();
                    const select = document.getElementById('matchId');
                    select.innerHTML = '<option value="">-- Choisir un match --</option>';
                    
                    if(matchs.erreur) {
                        select.innerHTML = '<option value="">Erreur de chargement</option>';
                        return;
                    }
                    
                    matchs.forEach(m => {
                        let option = document.createElement('option');
                        option.value = m.id;
                        option.text = `${m.nom} (${m.date})`;
                        select.appendChild(option);
                    });
                } catch (e) {
                    console.error("Erreur chargement matchs:", e);
                }
            }

            window.onload = chargerMatchs;

            async function analyserMatch() {
                const matchId = document.getElementById('matchId').value;
                if(!matchId) { alert("Veuillez sélectionner un match."); return; }

                document.getElementById('resultats').style.display = 'none';

                try {
                    const resC = await fetch(`/predict-corners/${matchId}`);
                    const dataC = await resC.json();
                    const resK = await fetch(`/predict-cards/${matchId}`);
                    const dataK = await resK.json();

                    if(dataC.erreur || dataK.erreur) { alert("Match introuvable."); return; }

                    currentMatchId = matchId;
                    currentDataC = dataC;
                    currentDataK = dataK;

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
                    let idInput = `cote_${cle}`;
                    
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

            async function sauvegarderPrediction() {
                if(!currentMatchId) return;
                
                try {
                    const response = await fetch(`/sauvegarder-prediction/${currentMatchId}`, { method: 'POST' });
                    const result = await response.json();
                    
                    if(result.erreur) {
                        alert("Erreur : " + result.erreur);
                    } else {
                        alert("✅ " + result.message);
                    }
                } catch (e) {
                    alert("Erreur de connexion.");
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/sauvegarder-prediction/{match_id_local}")
async def sauvegarder_prediction(match_id_local: int):
    if not SessionLocal: return {"erreur": "Pas de BDD"}
    db = SessionLocal()
    try:
        # On récupère les données qu'on vient de calculer
        dataC = await predict_corners(match_id_local)
        dataK = await predict_cards(match_id_local)
        
        if "erreur" in dataC or "erreur" in dataK:
            return {"erreur": "Impossible de calculer les probabilités."}

        # On enregistre les prédictions Corners
        p_corners = Prediction(
            match_id=match_id_local,
            marche="corners",
            ligne=9.5,
            probabilite_brute=dataC["analyse_corners"]["probabilite_plus_de_9_5"],
            probabilite_calibree=dataC["analyse_corners"]["probabilite_plus_de_9_5"],
            modele_version="poisson-v1.0",
            date_prediction=datetime.utcnow()
        )
        db.add(p_corners)

        # On enregistre les prédictions Cartons
        p_cartons = Prediction(
            match_id=match_id_local,
            marche="cartons",
            ligne=4.5,
            probabilite_brute=dataK["analyse_cartons"]["probabilite_plus_de_4_5"],
            probabilite_calibree=dataK["analyse_cartons"]["probabilite_plus_de_4_5"],
            modele_version="poisson-v1.0",
            date_prediction=datetime.utcnow()
        )
        db.add(p_cartons)

        db.commit()
        return {"message": "Analyse sauvegardée dans la base de données avec succès !"}
    except Exception as e:
        db.rollback()
        return {"erreur": str(e)}
    finally:
        db.close()
