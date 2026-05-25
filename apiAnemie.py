from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json, os
import numpy as np
import pandas as pd

app = FastAPI(title="IMAS AI — API Anémie")

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
    return {"message": "IMAS AI API — Anémie enfants < 5 ans"}

@app.get("/health")
def health():
    return {"status": "ok", "colonnes": len(COLONNES)}

@app.post("/predict")
def predict(data: PatientData):
    X         = preparer_input(data)
    pred_rf   = int(rf.predict(X)[0])
    proba_rf  = rf.predict_proba(X)[0]
    proba_ord = result_robust.predict(X).values[0]
    pred_ord  = int(np.argmax(proba_ord))
    return {
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






# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import joblib
# import json
# import numpy as np
# import pandas as pd
# import os

# app = FastAPI(title="IMAS AI — API Anémie")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ── Chargement des modèles ─────────────────────────────
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # ← ajouter ceci

# rf            = joblib.load(os.path.join(BASE_DIR, "random_forest_anemie.pkl"))  # ← modifier
# result_robust = joblib.load(os.path.join(BASE_DIR, "ordinal_anemie.pkl"))        # ← modifier

# with open(os.path.join(BASE_DIR, "colonnes_model.json")) as f: 

# # rf            = joblib.load("random_forest_anemie.pkl")
# # result_robust = joblib.load("ordinal_anemie.pkl")

# # with open("colonnes_model.json") as f:
#     COLONNES = json.load(f)

# LABELS = {0: "Pas anémie", 1: "Anémie légère", 2: "Anémie modérée/sévère"}


# # ── Schéma d'entrée ────────────────────────────────────
# class PatientData(BaseModel):
#     age_enfant_mois:     float
#     zscore_taille_age:   float
#     zscore_poids_taille: float
#     diarrhee:            int   # 0=Non  1=Oui
#     fievre:              int   # 0=Non  1=Oui
#     anemie_mere:         int   # 0=Aucune 1=Légère 2=Modérée 3=Sévère
#     milieu_residence:    int   # 0=Urbain 1=Rural
#     indice_richesse:     int   # 1=Très pauvre ... 5=Très riche
#     niveau_instruction:  int   # 0=Aucun 1=Primaire 2=Secondaire 3=Supérieur
#     age_mere:            float
#     BMI:                 float


# def preparer_input(data: PatientData) -> pd.DataFrame:
#     """
#     Encode les données exactement comme pendant l'entraînement.
#     Produit les 18 colonnes : age_enfant_mois, zscore_taille_age,
#     zscore_poids_taille, diarrhee, fievre, age_mere, BMI,
#     anemie_mere_0, anemie_mere_2, anemie_mere_3,
#     milieu_residence_0,
#     indice_richesse_1, indice_richesse_2, indice_richesse_4, indice_richesse_5,
#     niveau_instruction_0, niveau_instruction_1, niveau_instruction_2
#     """

#     # Variables continues directement
#     row = {
#         'age_enfant_mois':     data.age_enfant_mois,
#         'zscore_taille_age':   data.zscore_taille_age,
#         'zscore_poids_taille': data.zscore_poids_taille,
#         'diarrhee':            data.diarrhee,
#         'fievre':              data.fievre,
#         'age_mere':            data.age_mere,
#         'BMI':                 data.BMI,
#     }

#     # Encodage manuel one-hot (drop_first=True → référence = catégorie 1)
#     # anemie_mere : ref=1 → on crée _0, _2, _3
#     row['anemie_mere_0'] = 1 if data.anemie_mere == 0 else 0
#     row['anemie_mere_2'] = 1 if data.anemie_mere == 2 else 0
#     row['anemie_mere_3'] = 1 if data.anemie_mere == 3 else 0

#     # milieu_residence : ref=1 → on crée _0
#     row['milieu_residence_0'] = 1 if data.milieu_residence == 0 else 0

#     # indice_richesse : ref=3 → on crée _1, _2, _4, _5
#     row['indice_richesse_1'] = 1 if data.indice_richesse == 1 else 0
#     row['indice_richesse_2'] = 1 if data.indice_richesse == 2 else 0
#     row['indice_richesse_4'] = 1 if data.indice_richesse == 4 else 0
#     row['indice_richesse_5'] = 1 if data.indice_richesse == 5 else 0

#     # niveau_instruction : ref=3 → on crée _0, _1, _2
#     row['niveau_instruction_0'] = 1 if data.niveau_instruction == 0 else 0
#     row['niveau_instruction_1'] = 1 if data.niveau_instruction == 1 else 0
#     row['niveau_instruction_2'] = 1 if data.niveau_instruction == 2 else 0

#     # Construire le DataFrame dans l'ordre exact des colonnes
#     df = pd.DataFrame([row])[COLONNES].astype(float)
#     return df


# @app.get("/")
# def root():
#     return {"message": "IMAS AI API — Détection anémie enfants < 5 ans"}


# @app.get("/health")
# def health():
#     return {"status": "ok", "colonnes": len(COLONNES)}


# @app.post("/predict")
# def predict(data: PatientData):
#     X_input = preparer_input(data)

#     # Random Forest
#     pred_rf  = int(rf.predict(X_input)[0])
#     proba_rf = rf.predict_proba(X_input)[0]

#     # Régression Ordinale
#     proba_ord = result_robust.predict(X_input).values[0]
#     pred_ord  = int(np.argmax(proba_ord))

#     return {
#         "random_forest": {
#             "prediction":  pred_rf,
#             "label":       LABELS[pred_rf],
#             "probabilites": {
#                 "pas_anemie":    round(float(proba_rf[0]), 4),
#                 "leger":         round(float(proba_rf[1]), 4),
#                 "modere_severe": round(float(proba_rf[2]), 4),
#             }
#         },
#         "ordinal": {
#             "prediction":  pred_ord,
#             "label":       LABELS[pred_ord],
#             "probabilites": {
#                 "pas_anemie":    round(float(proba_ord[0]), 4),
#                 "leger":         round(float(proba_ord[1]), 4),
#                 "modere_severe": round(float(proba_ord[2]), 4),
#             }
#         },
#         "diagnostic_final": LABELS[pred_rf]
#     }