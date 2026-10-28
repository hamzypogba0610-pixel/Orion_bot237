from fastapi import FastAPI
import os
import httpx

app = FastAPI(title="ORION", version="0.1.0")

@app.get("/")
async def home():
    return {"message": "ORION est en ligne", "version": "0.1.0", "statut": "vivant"}

@app.get("/test-matchs/{date_recherche}")
async def test_matchs(date_recherche: str):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key: return {"erreur": "Clé introuvable"}
    url = f"https://v3.football.api-sports.io/fixtures?date={date_recherche}"
    headers = {"x-apisports-key": api_key}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    data = response.json()
    if data.get("results", 0) == 0: return {"message": "Aucun match."}
    
    matchs = []
    for m in data["response"][:5]:
        matchs.append({
            "id_match": m["fixture"]["id"],
            "competition": m["league"]["name"],
            "equipe_domicile": m["teams"]["home"]["name"],
            "equipe_exterieur": m["teams"]["away"]["name"],
            "score_final": f"{m['goals']['home']} - {m['goals']['away']}"
        })
    return matchs

# NOUVELLE PORTE : Uniquement les grands championnats
@app.get("/test-top-matchs/{date_recherche}")
async def test_top_matchs(date_recherche: str):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key: return {"erreur": "Clé introuvable"}

    url = f"https://v3.football.api-sports.io/fixtures?date={date_recherche}"
    headers = {"x-apisports-key": api_key}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        
    data = response.json()
    if data.get("results", 0) == 0: return {"message": "Aucun match."}

    # IDs des grands championnats : 39=Premier League, 140=La Liga, 135=Serie A, 61=Ligue 1, 78=Bundesliga
    top_leagues = [39, 140, 135, 61, 78]
    top_matchs = []

    for m in data["response"]:
        if m["league"]["id"] in top_leagues:
            top_matchs.append({
                "id_match": m["fixture"]["id"],
                "competition": m["league"]["name"],
                "equipe_domicile": m["teams"]["home"]["name"],
                "equipe_exterieur": m["teams"]["away"]["name"],
                "statut": m["fixture"]["status"]["short"],
                "score": f"{m['goals']['home']} - {m['goals']['away']}"
            })
            
    if not top_matchs:
        return {"message": "Aucun match des grands championnats trouvé pour cette date."}
        
    return top_matchs[:10] # On affiche jusqu'à 10 matchs

@app.get("/test-stats/{match_id}")
async def test_stats(match_id: int):
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key: return {"erreur": "Clé introuvable"}
    url = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={match_id}"
    headers = {"x-apisports-key": api_key}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    data = response.json()
    if data.get("results", 0) == 0: return {"message": "Aucune statistique pour ce match."}
    return data["response"]
