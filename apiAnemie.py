from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predict_clmm import predict_clmm
from pydantic import BaseModel
import joblib, json, os
import numpy as np
import pandas as pd

from sqlalchemy.orm import Session
from database import Consultation, get_db, init_db

# =========================
#  APP
# =========================
app = FastAPI(title="Anemia AI — API Anémie")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
#  MODELES
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
#  INIT DB
# =========================
@app.on_event("startup")
def startup():
    init_db()

# =========================
#  DATAFRAME SAFE
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
#  INPUT MODEL
# =========================
class PatientData(BaseModel):
    sexe_enfant:        int
    age_enfant_mois:    float
    diarrhee:           int
    fievre:             int
    deparasitage:       int
    age_mere:           float
    anemie_mere:        int
    indice_richesse:    int
    niveau_instruction: int
    milieu_residence:   int
    region:             int
    type_allaitement:   int

    nom_praticien:   str | None = None
    structure_sante: str | None = None
    notes:           str | None = None

# =========================
#  PREPROCESSING
# =========================
def preparer_input(data: PatientData) -> pd.DataFrame:
    row = {
        "sexe_enfant":        data.sexe_enfant,
        "age_enfant_mois":    data.age_enfant_mois,
        "diarrhee":           data.diarrhee,
        "fievre":             data.fievre,
        "deparasitage":       data.deparasitage,
        "age_mere":           data.age_mere,
        "anemie_mere":        data.anemie_mere,
        "indice_richesse":    data.indice_richesse,
        "niveau_instruction": data.niveau_instruction,
        "milieu_residence":   data.milieu_residence,
        "region":             data.region,
        "type_allaitement":   data.type_allaitement,
    }
    df = pd.DataFrame([row])
    return df[COLONNES].astype(float)

# =========================
#  ROOT
# =========================
@app.get("/")
def root():
    return {"message": "Anemia AI API OK"}

@app.get("/health")
def health():
    return {"status": "ok", "features": len(COLONNES)}

# =========================
#  STATS ANÉMIE
# =========================
@app.get("/stats/anemie")
def stats_anemie(db: Session = Depends(get_db)):
    df = get_dataframe(db)
    if df.empty:
        return {
            "labels":  ["Pas anémie", "Légère", "Modérée/Sévère"],
            "data":    [0, 0, 0],
            "percent": [0, 0, 0],
            "colors":  ["#2e7d32", "#e65100", "#c62828"]
        }
    vc    = df["prediction_rf"].value_counts().sort_index()
    total = len(df)
    return {
        "labels":  ["Pas anémie", "Légère", "Modérée/Sévère"],
        "data":    [int(vc.get(i, 0)) for i in range(3)],
        "percent": [round(vc.get(i, 0) / total * 100, 1) for i in range(3)],
        "colors":  ["#2e7d32", "#e65100", "#c62828"]
    }
 
# =========================
#  FEATURES
# =========================   

@app.get("/stats/features")
def stats_features():

    names = COLONNES
    importances = rf.feature_importances_

    data = [
        {
            "name": n,
            "pct": round(float(i * 100), 2)
        }
        for n, i in zip(names, importances)
    ]

    data.sort(key=lambda x: x["pct"], reverse=True)
    return data


# =========================
#  Tranche dage
# =========================

@app.get("/stats/age")
def stats_age(db: Session = Depends(get_db)):

    df = get_dataframe(db)
    if df.empty:
        return []

    # df = df.dropna(subset=["age_enfant_mois", "prediction_rf"])  # ← sécurité ajoutée
    # df["tranche"] = pd.cut(
    #     df["age_enfant_mois"],
    #     bins=[0, 6, 12, 24, 36, 60],
    #     labels=["0-6", "6-12", "12-24", "24-36", "36-60"]
    # )

    # result = df.groupby("tranche")["prediction_rf"].mean().reset_index()

    # return [
    #     {
    #         "label": r["tranche"],
    #         "pct": round(r["prediction_rf"] * 100, 2)
    #     }
    #     for _, r in result.iterrows()
    # ]
    
    df["tranche"] = pd.cut(
    df["age_enfant_mois"],
    bins=[0, 6, 12, 24, 36, 60],
    labels=["0-6", "6-12", "12-24", "24-36", "36-60"],
    include_lowest=True
)

    result = df.groupby("tranche", observed=True)["prediction_rf"].mean()
    
# =========================
#  PAR REGION
# =========================
    
@app.get("/stats/anemie/region")
def stats_anemie_region(db: Session = Depends(get_db)):

    df = get_dataframe(db)

    if df.empty:
        return []

    # moyenne de l'anémie par région
    # (0,1,2 → on convertit en "niveau moyen")
    df = df.dropna(subset=["region", "prediction_rf"])
    # grouped = df.groupby("region")["prediction_rf"].mean().reset_index()

    # return [
    #     {
    #         "region": int(row["region"]),
    #         "pct": round(float(row["prediction_rf"] / 2 * 100), 2)  # normalisation 0–100%
    #     }
    #     for _, row in grouped.iterrows()
    # ]
    
    grouped = df.groupby("region")["prediction_rf"].mean()

    return [
        {
            "region": int(region),
            "pct": round(value * 100, 1)
        }
        for region, value in grouped.items()
    ]
    
# =========================
#  PAR SEXE
# =========================
    
@app.get("/stats/anemie-sexe")
def stats_anemie_sexe(db: Session = Depends(get_db)):
    df = get_dataframe(db)

    if df.empty:
        return {
            "labels": ["Garçons", "Filles"],
            "percent": [0, 0],
            "colors": ["#42a5f5", "#ec407a"]
        }

    # total = len(df)


    # 0 = fille, 1 = garçon (adapte si besoin)
    sexe_stats = df.groupby("sexe_enfant")["prediction_rf"].mean()
    # df.groupby("sexe_enfant")["prediction_rf"].apply(
    #     lambda x: (x > 0).sum() / len(x) * 100
    # )

    return {
        "labels": ["Filles", "Garçons"],
        "percent": [
            round(sexe_stats.get(0, 0), 1),
            round(sexe_stats.get(1, 0), 1)
        ],
        "colors": ["#ec407a", "#42a5f5"]
    }

# =========================
#  DASHBOARD
# =========================
@app.get("/stats/dashboard")
def dashboard(db: Session = Depends(get_db)):
    df = get_dataframe(db)
    if df.empty:
        return {
            "total_consultations": 0,
            "anemie": {"0": 0, "1": 0, "2": 0}
        }
    return {
        "total_consultations": len(df),
        "anemie": {
            "0": int((df["prediction_rf"] == 0).sum()),
            "1": int((df["prediction_rf"] == 1).sum()),
            "2": int((df["prediction_rf"] == 2).sum())
        }
    }

# =========================
#  PRÉDICTION
# =========================
@app.post("/predict")
def predict(data: PatientData, db: Session = Depends(get_db)):

    # ── Random Forest ──────────────────────────────────
    X        = preparer_input(data)
    pred_rf  = int(rf.predict(X)[0])
    proba_rf = rf.predict_proba(X)[0]

    # ── CLMM R (coefficients Python) ───────────────────
    
    try:
        clmm_result = predict_clmm(data.model_dump())
        pred_ord = clmm_result["prediction"]
        proba_ord = clmm_result["probabilites"]
    except Exception:
        pred_ord = pred_rf
        proba_ord = {"pas_anemie": 0, "leger": 0, "modere_severe": 0}
        
    # clmm_result = predict_clmm(data.model_dump())
    # pred_ord    = clmm_result["prediction"]
    # proba_ord   = clmm_result["probabilites"]

    # ── Sauvegarde BDD ─────────────────────────────────
    consultation = Consultation(
        **data.model_dump(),
        sexe_enfant=        data.sexe_enfant,
        age_enfant_mois=    data.age_enfant_mois,
        diarrhee=           data.diarrhee,
        fievre=             data.fievre,
        deparasitage=       data.deparasitage,
        age_mere=           data.age_mere,
        anemie_mere=        data.anemie_mere,
        indice_richesse=    data.indice_richesse,
        niveau_instruction= data.niveau_instruction,
        milieu_residence=   data.milieu_residence,
        region=             data.region,
        type_allaitement=   data.type_allaitement,
        
        
        prediction_rf=      pred_rf,
        label_rf=           LABELS[pred_rf],
        prediction_ord=     pred_ord,
        diagnostic_final=   LABELS[pred_rf],
        nom_praticien=      data.nom_praticien,
        structure_sante=    data.structure_sante,
        notes=              data.notes,
    )
    db.add(consultation)
    db.commit()
    db.refresh(consultation)

    # ── Réponse ────────────────────────────────────────
    return {
        "id": consultation.id,

        "random_forest": {
            "prediction": pred_rf,
            "label":      LABELS[pred_rf],
            "probabilites": {
                "classe_0": round(float(proba_rf[0]), 4),
                "classe_1": round(float(proba_rf[1]), 4),
                "classe_2": round(float(proba_rf[2]), 4),
            }
        },

        "ordinal": {
            "prediction": pred_ord,
            "label":      LABELS[pred_ord],
            "probabilites": {
                "pas_anemie":    proba_ord["pas_anemie"],
                "leger":         proba_ord["leger"],
                "modere_severe": proba_ord["modere_severe"],
            }
        },

        "diagnostic_final": LABELS[pred_rf]
    }

# =========================
#  CONSULTATIONS
# =========================
@app.get("/consultations")
def liste_consultations(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    total = db.query(Consultation).count()
    items = db.query(Consultation) \
        .order_by(Consultation.created_at.desc()) \
        .offset(skip) \
        .limit(limit) \
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