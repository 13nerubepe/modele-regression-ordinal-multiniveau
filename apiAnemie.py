from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json, os
import numpy as np
import pandas as pd

from sqlalchemy.orm import Session
from database import Consultation, get_db, init_db

app = FastAPI(title="Anemia AI — API Anémie")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
rf            = joblib.load(os.path.join(BASE_DIR, "rf.pkl"))
result_robust = joblib.load(os.path.join(BASE_DIR, "result_robust.pkl"))

with open(os.path.join(BASE_DIR, "colonnes_model.json")) as f:
    COLONNES = json.load(f)

LABELS = {0: "Pas anémie", 1: "Anémie légère", 2: "Anémie modérée/sévère"}


# ── Créer les tables au démarrage ──────────────────────
@app.on_event("startup")
def startup():
    init_db()


class PatientData(BaseModel):
    age_enfant_mois:     float
    zscore_taille_age:   float
    zscore_poids_taille: float
    diarrhee:            int
    fievre:              int
    anemie_mere:         int
    milieu_residence:    int
    indice_richesse:     int
    niveau_instruction:  int
    age_mere:            float
    BMI:                 float

    # Champs optionnels
    nom_praticien:       str | None = None
    structure_sante:     str | None = None
    notes:               str | None = None


def preparer_input(data: PatientData) -> pd.DataFrame:
    row = {
        'age_enfant_mois':      data.age_enfant_mois,
        'zscore_taille_age':    data.zscore_taille_age,
        'zscore_poids_taille':  data.zscore_poids_taille,
        'diarrhee':             float(data.diarrhee),
        'fievre':               float(data.fievre),
        'age_mere':             data.age_mere,
        'BMI':                  data.BMI,
        'anemie_mere_0':        1.0 if data.anemie_mere == 0 else 0.0,
        'anemie_mere_2':        1.0 if data.anemie_mere == 2 else 0.0,
        'anemie_mere_3':        1.0 if data.anemie_mere == 3 else 0.0,
        'milieu_residence_0':   1.0 if data.milieu_residence == 0 else 0.0,
        'indice_richesse_1':    1.0 if data.indice_richesse == 1 else 0.0,
        'indice_richesse_2':    1.0 if data.indice_richesse == 2 else 0.0,
        'indice_richesse_4':    1.0 if data.indice_richesse == 4 else 0.0,
        'indice_richesse_5':    1.0 if data.indice_richesse == 5 else 0.0,
        'niveau_instruction_0': 1.0 if data.niveau_instruction == 0 else 0.0,
        'niveau_instruction_1': 1.0 if data.niveau_instruction == 1 else 0.0,
        'niveau_instruction_2': 1.0 if data.niveau_instruction == 2 else 0.0,
    }
    return pd.DataFrame([row])[COLONNES].astype(float)


@app.get("/")
def root():
    return {"message": "Anemia AI API — Anémie enfants < 5 ans"}

@app.get("/health")
def health():
    return {"status": "ok", "colonnes": len(COLONNES)}

@app.post("/predict")
def predict(data: PatientData, db: Session = Depends(get_db)):
    X         = preparer_input(data)
    pred_rf   = int(rf.predict(X)[0])
    proba_rf  = rf.predict_proba(X)[0]
    proba_ord = result_robust.predict(X).values[0]
    pred_ord  = int(np.argmax(proba_ord))

     # ── Sauvegarder en base ────────────────────────────
    consultation = Consultation(
        age_enfant_mois     = data.age_enfant_mois,
        zscore_taille_age   = data.zscore_taille_age,
        zscore_poids_taille = data.zscore_poids_taille,
        diarrhee            = data.diarrhee,
        fievre              = data.fievre,
        anemie_mere         = data.anemie_mere,
        milieu_residence    = data.milieu_residence,
        indice_richesse     = data.indice_richesse,
        niveau_instruction  = data.niveau_instruction,
        age_mere            = data.age_mere,
        bmi                 = data.BMI,
        prediction_rf       = pred_rf,
        label_rf            = LABELS[pred_rf],
        proba_pas_anemie_rf = round(float(proba_rf[0]), 4),
        proba_leger_rf      = round(float(proba_rf[1]), 4),
        proba_severe_rf     = round(float(proba_rf[2]), 4),
        prediction_ord      = pred_ord,
        label_ord           = LABELS[pred_ord],
        diagnostic_final    = LABELS[pred_rf],
        nom_praticien       = data.nom_praticien,
        structure_sante     = data.structure_sante,
        notes               = data.notes,
    )
    db.add(consultation)
    db.commit()
    db.refresh(consultation)


    return {
         "id": consultation.id,
        "random_forest": {
            "prediction":  pred_rf,
            "label":       LABELS[pred_rf],
            "probabilites": {
                "pas_anemie":    round(float(proba_rf[0]), 4),
                "leger":         round(float(proba_rf[1]), 4),
                "modere_severe": round(float(proba_rf[2]), 4),
            }
        },
        "ordinal": {
            "prediction":  pred_ord,
            "label":       LABELS[pred_ord],
            "probabilites": {
                "pas_anemie":    round(float(proba_ord[0]), 4),
                "leger":         round(float(proba_ord[1]), 4),
                "modere_severe": round(float(proba_ord[2]), 4),
            }
        },
        "diagnostic_final": LABELS[pred_rf]
    }



    # ── Endpoints pour Angular ─────────────────────────────

# ── GET /consultations → appelé par getHistorique() Angular ──
@app.get("/consultations")
def liste_consultations(
    skip:  int = 0,
    limit: int = 50,
    db:    Session = Depends(get_db)
):
    total = db.query(Consultation).count()
    items = db.query(Consultation) \
              .order_by(Consultation.created_at.desc()) \
              .offset(skip).limit(limit).all()
    return {"total": total, "items": items}


# ── GET /consultations/{id} → appelé par getConsultation() Angular ──
@app.get("/consultations/{id}")
def detail_consultation(id: int, db: Session = Depends(get_db)):
    c = db.query(Consultation).filter(Consultation.id == id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Consultation introuvable")
    return c

# ── DELETE /consultations/{id} → appelé par supprimerConsultation() Angular ──
@app.delete("/consultations/{id}")
def supprimer_consultation(id: int, db: Session = Depends(get_db)):
    from fastapi import HTTPException
    c = db.query(Consultation).filter(Consultation.id == id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Consultation introuvable")
    db.delete(c)
    db.commit()
    return {"message": "Supprimé"}

