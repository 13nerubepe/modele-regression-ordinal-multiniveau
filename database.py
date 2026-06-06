from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Le fichier .db sera créé automatiquement ici
DATABASE_URL = "sqlite:///./imas_anemie.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Consultation(Base):
    __tablename__ = "consultations"

    id                   = Column(Integer, primary_key=True, index=True)
    created_at           = Column(DateTime, default=datetime.utcnow)
    age_enfant_mois      = Column(Float)
    zscore_taille_age    = Column(Float)
    zscore_poids_taille  = Column(Float)
    diarrhee             = Column(Integer)
    fievre               = Column(Integer)
    anemie_mere          = Column(Integer)
    milieu_residence     = Column(Integer)
    indice_richesse      = Column(Integer)
    niveau_instruction   = Column(Integer)
    age_mere             = Column(Float)
    bmi                  = Column(Float)
    prediction_rf        = Column(Integer)
    label_rf             = Column(String)
    proba_pas_anemie_rf  = Column(Float)
    proba_leger_rf       = Column(Float)
    proba_severe_rf      = Column(Float)
    prediction_ord       = Column(Integer)
    label_ord            = Column(String)
    diagnostic_final     = Column(String)
    nom_praticien        = Column(String, nullable=True)
    structure_sante      = Column(String, nullable=True)
    notes                = Column(String, nullable=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)