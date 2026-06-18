from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

import joblib
import json
import os
import numpy as np
import pandas as pd

from database import Consultation, get_db, init_db
from predict_clmm import predict_clmm

# =========================
# APP
# =========================
app = FastAPI(title="Anemia AI — API Anémie")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELE
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf = joblib.load(os.path.join(BASE_DIR, "rf.pkl"))

with open(os.path.join(BASE_DIR, "colonnes_model.json")) as f:
    COLONNES = json.load(f)

LABELS = {
    0: "Pas anémie",
    1: "Anémie légère",
    2: "Anémie modérée/sévère"
}

# =========================
# INIT DB
# =========================
@app.on_event("startup")
def startup():
    init_db()

# =========================
# INPUT MODEL
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
# PREPROCESSING SAFE
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
    df = df.reindex(columns=COLONNES)

    # sécurité anti-crash ML
    df = df.fillna(0)

    return df.astype(float)

# =========================
# GET DATAFRAME SAFE
# =========================
def get_dataframe(db: Session):
    rows = db.query(Consultation).all()

    if not rows:
        return pd.DataFrame(columns=["prediction_rf", "prediction_ord"])

    df = pd.DataFrame([r.__dict__ for r in rows])

    if "_sa_instance_state" in df.columns:
        df = df.drop(columns=["_sa_instance_state"])

    return df

# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"message": "Anemia AI API OK"}

@app.get("/health")
def health():
    return {"status": "ok", "features": len(COLONNES)}

# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(data: PatientData, db: Session = Depends(get_db)):

    try:
        # ================= RF =================
        X = preparer_input(data)

        pred_rf = int(rf.predict(X)[0])
        proba_rf = rf.predict_proba(X)[0]

        pred_rf = max(0, min(pred_rf, 2))

        # ================= CLMM SAFE =================
        try:
            clmm_result = predict_clmm(data.model_dump())

            pred_ord = clmm_result.get("prediction", pred_rf)

            try:
                pred_ord = int(float(pred_ord))
            except:
                pred_ord = pred_rf

            proba_ord = clmm_result.get("probabilites", {})

        except Exception as e:
            print("CLMM ERROR:", str(e))
            pred_ord = pred_rf
            proba_ord = {
                "pas_anemie": 0.0,
                "leger": 0.0,
                "modere_severe": 0.0
            }

        pred_ord = max(0, min(pred_ord, 2))

        # ================= SAVE DB =================
        consultation = Consultation(
            sexe_enfant=data.sexe_enfant,
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
            label_rf=LABELS.get(pred_rf),
            prediction_ord=pred_ord,
            diagnostic_final=LABELS.get(pred_ord),

            nom_praticien=data.nom_praticien,
            structure_sante=data.structure_sante,
            notes=data.notes,
        )

        db.add(consultation)
        db.commit()
        db.refresh(consultation)

        return {
            "id": consultation.id,

            "random_forest": {
                "prediction": pred_rf,
                "label": LABELS.get(pred_rf),
                "probabilites": {
                    "classe_0": round(float(proba_rf[0]), 4),
                    "classe_1": round(float(proba_rf[1]), 4),
                    "classe_2": round(float(proba_rf[2]), 4),
                }
            },

            "ordinal": {
                "prediction": pred_ord,
                "label": LABELS.get(pred_ord),
                "probabilites": {
                    "pas_anemie": proba_ord.get("pas_anemie", 0),
                    "leger": proba_ord.get("leger", 0),
                    "modere_severe": proba_ord.get("modere_severe", 0),
                }
            },

            "diagnostic_final": LABELS.get(pred_ord)
        }

    except Exception as e:
        print("GLOBAL ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# STATS
# =========================
@app.get("/stats/anemie")
def stats_anemie(db: Session = Depends(get_db)):
    df = get_dataframe(db)

    if df.empty:
        return {
            "labels": ["Pas", "Légère", "Sévère"],
            "data": [0, 0, 0],
            "percent": [0, 0, 0]
        }

    vc = df["prediction_rf"].value_counts().sort_index()
    total = len(df)

    return {
        "labels": ["Pas anémie", "Légère", "Sévère"],
        "data": [int(vc.get(i, 0)) for i in range(3)],
        "percent": [round(vc.get(i, 0)/total*100, 1) for i in range(3)]
    }

@app.get("/stats/dashboard")
def dashboard(db: Session = Depends(get_db)):
    df = get_dataframe(db)

    if df.empty:
        return {"total": 0, "anemie": {"0": 0, "1": 0, "2": 0}}

    return {
        "total": len(df),
        "anemie": {
            "0": int((df["prediction_rf"] == 0).sum()),
            "1": int((df["prediction_rf"] == 1).sum()),
            "2": int((df["prediction_rf"] == 2).sum())
        }
    }

# =========================
# CONSULTATIONS
# =========================
@app.get("/consultations")
def liste_consultations(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    total = db.query(Consultation).count()
    items = db.query(Consultation)\
        .order_by(Consultation.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

    return {"total": total, "items": items}


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