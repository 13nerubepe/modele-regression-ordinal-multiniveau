from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json, os
import numpy as np
import pandas as pd

from sqlalchemy.orm import Session
from database import Consultation, get_db, init_db

# =========================
# 🚀 APP
# =========================
app = FastAPI(title="Anemia AI — API Anémie")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 📦 MODELES
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf = joblib.load(os.path.join(BASE_DIR, "rf.pkl"))
result_robust = joblib.load(os.path.join(BASE_DIR, "result_robust.pkl"))

with open(os.path.join(BASE_DIR, "colonnes_model.json")) as f:
    COLONNES = json.load(f)

LABELS = {
    0: "Pas anémie",
    1: "Anémie légère",
    2: "Anémie modérée/sévère"
}

# =========================
# 🗄️ INIT DB
# =========================
@app.on_event("startup")
def startup():
    init_db()

# =========================
# 📌 INPUT MODEL
# =========================
class PatientData(BaseModel):
    sexe_enfant: int
    age_enfant_mois: float
    diarrhee: int
    fievre: int
    deparasitage: int
    age_mere: float
    anemie_mere: int
    indice_richesse: int
    niveau_instruction: int
    milieu_residence: int
    region: int
    type_allaitement: int

    nom_praticien: str | None = None
    structure_sante: str | None = None
    notes: str | None = None

# =========================
# 📊 PREPROCESSING
# =========================
def preparer_input(data: PatientData) -> pd.DataFrame:

    row = {
        "sexe_enfant": data.sexe_enfant,
        "age_enfant_mois": data.age_enfant_mois,
        "diarrhee": data.diarrhee,
        "fievre": data.fievre,
        "deparasitage": data.deparasitage,
        "age_mere": data.age_mere,
        "anemie_mere": data.anemie_mere,
        "indice_richesse": data.indice_richesse,
        "niveau_instruction": data.niveau_instruction,
        "milieu_residence": data.milieu_residence,
        "region": data.region,
        "type_allaitement": data.type_allaitement,
    }

    df = pd.DataFrame([row])
    return df[COLONNES].astype(float)

# =========================
# 🏠 ROOT
# =========================
@app.get("/")
def root():
    return {"message": "Anemia AI API OK"}

@app.get("/health")
def health():
    return {"status": "ok", "features": len(COLONNES)}

# =========================
# 🤖 PREDICTION (ML CORE)
# =========================
@app.post("/predict")
def predict(data: PatientData, db: Session = Depends(get_db)):

    X = preparer_input(data)

    # =========================
    #  RANDOM FOREST
    # =========================
    pred_rf = int(rf.predict(X)[0])
    proba_rf = rf.predict_proba(X)[0]

    # =========================
    #  ORDINAL MODEL
    # =========================
    # proba_ord = result_robust.predict(X).values[0]
    # pred_ord = int(np.argmax(proba_ord))

    # =========================
    #  SAVE DATABASE
    # =========================
    consultation = Consultation(
        age_enfant_mois=data.age_enfant_mois,
        diarrhee=data.diarrhee,
        fievre=data.fievre,
        deparasitage=data.deparasitage,
        age_mere=data.age_mere,
        anemie_mere=data.anemie_mere,
        indice_richesse=data.indice_richesse,
        niveau_instruction=data.niveau_instruction,
        milieu_residence=data.milieu_residence,
        region=data.region,
        type_allaitement=data.type_allaitement,

        prediction_rf=pred_rf,
        label_rf=LABELS[pred_rf],

        # prediction_ord=pred_ord,
        # label_ord=LABELS[pred_ord],

        diagnostic_final=LABELS[pred_rf],

        nom_praticien=data.nom_praticien,
        structure_sante=data.structure_sante,
        notes=data.notes,
    )

    db.add(consultation)
    db.commit()
    db.refresh(consultation)

    # =========================
    #  RESPONSE
    # =========================
    return {
        "id": consultation.id,

        "random_forest": {
            "prediction": pred_rf,
            "label": LABELS[pred_rf],
            "probabilites": {
                "classe_0": round(float(proba_rf[0]), 4),
                "classe_1": round(float(proba_rf[1]), 4),
                "classe_2": round(float(proba_rf[2]), 4),
            }
        },

        # "ordinal": {
        #     "prediction": pred_ord,
        #     "label": LABELS[pred_ord],
        #     "probabilites": {
        #         "classe_0": round(float(proba_ord[0]), 4),
        #         "classe_1": round(float(proba_ord[1]), 4),
        #         "classe_2": round(float(proba_ord[2]), 4),
        #     }
        # },

        "diagnostic_final": LABELS[pred_rf]
    }

# =========================
# 📱 IONIC / ANGULAR ENDPOINTS
# =========================

@app.get("/consultations")
def liste_consultations(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):

    total = db.query(Consultation).count()

    items = db.query(Consultation) \
        .order_by(Consultation.created_at.desc()) \
        .offset(skip) \
        .limit(limit) \
        .all()

    return {
        "total": total,
        "items": items
    }


@app.get("/consultations/{id}")
def detail_consultation(id: int, db: Session = Depends(get_db)):

    c = db.query(Consultation).filter(Consultation.id == id).first()

    if not c:
        raise HTTPException(status_code=404, detail="Consultation introuvable")

    return c


@app.delete("/consultations/{id}")
def supprimer_consultation(id: int, db: Session = Depends(get_db)):

    c = db.query(Consultation).filter(Consultation.id == id).first()

    if not c:
        raise HTTPException(status_code=404, detail="Consultation introuvable")

    db.delete(c)
    db.commit()

    return {"message": "Supprimé avec succès"}