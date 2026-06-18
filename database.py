from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# =========================
# DATABASE CONFIG
# =========================
DATABASE_URL = "sqlite:///./database.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# =========================
# TABLE CONSULTATION
# =========================
class Consultation(Base):
    __tablename__ = "consultations"

    # ID
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # =========================
    # INPUT PATIENT
    # =========================
    sexe_enfant = Column(Integer, nullable=False)
    age_enfant_mois = Column(Float, nullable=False)
    diarrhee = Column(Integer, nullable=False)
    fievre = Column(Integer, nullable=False)
    deparasitage = Column(Integer, nullable=False)

    age_mere = Column(Float, nullable=False)
    anemie_mere = Column(Integer, nullable=False)

    indice_richesse = Column(Integer, nullable=False)
    niveau_instruction = Column(Integer, nullable=False)
    milieu_residence = Column(Integer, nullable=False)
    region = Column(Integer, nullable=False)
    type_allaitement = Column(Integer, nullable=False)

    # =========================
    # PREDICTIONS
    # =========================
    prediction_rf = Column(Integer, nullable=False)
    prediction_ord = Column(Integer, nullable=False)
    label_rf = Column(String, nullable=False)
    diagnostic_final = Column(String, nullable=False)

    # =========================
    # METADATA (OPTIONNEL)
    # =========================
    nom_praticien = Column(String, nullable=True)
    structure_sante = Column(String, nullable=True)
    notes = Column(String, nullable=True)


# =========================
# INIT DB
# =========================
def init_db():
    Base.metadata.create_all(bind=engine)

# =========================
# SESSION
# =========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()