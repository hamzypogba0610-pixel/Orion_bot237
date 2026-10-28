from fastapi import FastAPI
import os
import httpx

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

# NOUVEAU : On met la date dans l'adresse !
@app.get("/test-matchs/{date_recherche}")
async def test_matchs(date_recherche: str):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key: return {"erreur": "Clé introuvable"}

    # L'adresse utilise la date qu'on tape dans le navigateur
    url = f"https://v3.football.api-sports.io/fixtures?date={date_recherche}"
    headers = {"x-apisports-key": api_key}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        
    data = response.json()
    
    if data.get("results", 0) == 0:
        return {"message": f"Aucun match trouvé pour la date {date_recherche}."}
        
    matchs_propres = []
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
