from fastapi import FastAPI
import os
import httpx
from datetime import date, timedelta

app = FastAPI(title="ORION", version="0.1.0")

@app.get("/")
async def home():
    return {"message": "ORION est en ligne", "version": "0.1.0", "statut": "vivant"}

@app.get("/test-api")
async def test_api():
    return {"message": "Ma nouvelle porte fonctionne !", "statut": "prêt pour les API"}

@app.get("/test-api-football")
async def test_api_football():
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key: return {"erreur": "Clé introuvable"}
    url = "https://v3.football.api-sports.io/status"
    headers = {"x-apisports-key": api_key}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    return response.json()

@app.get("/test-matchs")
async def test_matchs():
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key: return {"erreur": "Clé introuvable"}

    # On regarde les matchs d'il y a 2 jours
    date_recherche = (date.today() - timedelta(days=2)).isoformat()
    url = f"https://v3.football.api-sports.io/fixtures?date={date_recherche}"
    headers = {"x-apisports-key": api_key}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        
    data = response.json()
    
    if data.get("results", 0) == 0:
        return {"message": "Aucun match trouvé ce jour-là."}
        
    matchs_propres = []
    # On prend 5 matchs au hasard
    for m in data["response"][:5]:
        fiche = {
            "id_match": m["fixture"]["id"],
            "competition": m["league"]["name"],
            "equipe_domicile": m["teams"]["home"]["name"],
            "equipe_exterieur": m["teams"]["away"]["name"]
        }
        matchs_propres.append(fiche)
    return matchs_propres

@app.get("/test-stats/{match_id}")
async def test_stats(match_id: int):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key: return {"erreur": "Clé introuvable"}

    url = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={match_id}"
    headers = {"x-apisports-key": api_key}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        
    data = response.json()
    
    if data.get("results", 0) == 0:
        return {"message": "Aucune statistique pour ce match."}

    return data["response"]
