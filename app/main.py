import os
import httpx
from fastapi import FastAPI

app = FastAPI(title="ORION", version="0.1.0")

@app.get("/")
async def home():
    return {
        "message": "ORION est en ligne",
        "version": "0.1.0",
        "statut": "vivant"
    }

@app.get("/test-api")
async def test_api():
    return {
        "message": "Ma nouvelle porte fonctionne !",
        "statut": "prêt pour les API"
    }

@app.get("/test-api-football")
async def test_api_football():
    # 1. On va chercher la clé dans le coffre-fort Render
    api_key = os.getenv("API_FOOTBALL_KEY")
    
    # 2. On vérifie que la clé est bien là
    if not api_key:
        return {"erreur": "Clé introuvable dans le coffre-fort !"}
    
    # 3. On prépare l'appel téléphonique
    url = "https://v3.football.api-sports.io/status"
    headers = {"x-apisports-key": api_key}
    
    # 4. On appelle l'API
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        
    # 5. On renvoie la réponse
    return response.json()
