from fastapi import FastAPI

app = FastAPI(title="ORION", version="0.1.0")

@app.get("/")
async def home():
    return {
        "message": "ORION est en ligne",
        "version": "0.1.0",
        "statut": "vivant"
    }
