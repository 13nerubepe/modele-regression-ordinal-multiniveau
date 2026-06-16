import numpy as np

LABELS = {
    0: "Pas anémie",
    1: "Anémie légère",
    2: "Anémie modérée/sévère"
}

CLMM_COEFS = {
    "sexe_enfant":         0.1152,
    "age_enfant_mois":    -0.0276,
    "diarrhee":            0.0236,
    "fievre":              0.2706,
    "deparasitage":       -0.1013,
    "age_mere":           -0.0057,
    "anemie_mere":         0.4052,
    "indice_richesse":    -0.1702,
    "niveau_instruction": -0.1287,
    "milieu_residence":    0.1270,
    "region":              0.0140,
    "type_allaitement":    0.0962,
}

THRESHOLDS = {"0|1": -1.4410, "1|2": -0.2179}


def predict_clmm(data: dict) -> dict:
    eta = sum(
        CLMM_COEFS[col] * float(data[col])  # data[col] car c'est un dict
        for col in CLMM_COEFS
    )
    t1, t2 = THRESHOLDS["0|1"], THRESHOLDS["1|2"]

    p0 = 1.0 / (1.0 + np.exp(-(t1 - eta)))
    p1 = 1.0 / (1.0 + np.exp(-(t2 - eta))) - p0
    p2 = max(0.0, 1.0 - p0 - p1)

    proba = [p0, p1, p2]
    pred  = int(np.argmax(proba))

    return {
        "prediction":  pred,
        "label":       LABELS[pred],
        "probabilites": {
            "pas_anemie":    round(p0, 4),
            "leger":         round(max(0.0, p1), 4),
            "modere_severe": round(p2, 4),
        }
    }