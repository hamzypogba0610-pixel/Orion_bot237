from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Competition(Base):
    __tablename__ = "competitions"
    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, index=True)
    pays = Column(String)
    id_source = Column(Integer, unique=True) # L'ID de l'API (ex: 39 pour Premier League)

class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True, index=True)
    competition_id = Column(Integer, ForeignKey("competitions.id"))
    date_heure = Column(DateTime)
    equipe_domicile = Column(String)
    equipe_exterieur = Column(String)
    statut = Column(String) # ex: "FT", "NS" (Not Started)
    id_source_apifootball = Column(Integer, unique=True) # L'ID du match dans l'API
    score_domicile = Column(Integer, nullable=True)
    score_exterieur = Column(Integer, nullable=True)
    logo_domicile = Column(String, nullable=True)
    logo_exterieur = Column(String, nullable=True)

class MatchStat(Base):
    __tablename__ = "match_stats"
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    source_api = Column(String) # ex: "api-football"
    corners_domicile = Column(Integer, nullable=True)
    corners_exterieur = Column(Integer, nullable=True)
    cartons_jaunes_domicile = Column(Integer, nullable=True)
    cartons_jaunes_exterieur = Column(Integer, nullable=True)
    cartons_rouges_domicile = Column(Integer, nullable=True)
    cartons_rouges_exterieur = Column(Integer, nullable=True)
    statut_validation = Column(String, default="manquant") # "confirmé", "manquant", "divergent"
    date_recuperation = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    marche = Column(String) # "corners" ou "cartons"
    ligne = Column(Float) # ex: 9.5
    probabilite_brute = Column(Float)
    probabilite_calibree = Column(Float, nullable=True)
    modele_version = Column(String)
    meteo = Column(String, nullable=True) # Saisie utilisateur
    style_adversaire = Column(String, nullable=True) # Saisie utilisateur
    arbitre = Column(String, nullable=True)
    arbitre_profil = Column(String, nullable=True)
    date_prediction = Column(DateTime, default=datetime.utcnow)

class UserOdd(Base):
    __tablename__ = "user_odds"
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    bookmaker = Column(String)
    marche_saisi = Column(String)
    ligne_saisie = Column(Float)
    cote_decimale = Column(Float)
    date_saisie = Column(DateTime, default=datetime.utcnow)

class Decision(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    user_odds_id = Column(Integer, ForeignKey("user_odds.id"), nullable=True)
    type_decision = Column(String) # "principale", "alternative", "rejet"
    ev_calculee = Column(Float, nullable=True)
    raison_decision = Column(String)
    date_decision = Column(DateTime, default=datetime.utcnow)
