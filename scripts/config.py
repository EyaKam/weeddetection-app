"""
=============================================================
  config.py — Configuration globale du projet
  ← Toutes les valeurs à modifier sont ici
=============================================================
"""

# ── Chemins ──────────────────────────────────────────────────
PATHS = {
    "model_global"  : "./models/best (5).pt",
    "model_stages" : "./models/best0.827modele2.pt",  # optionnel
    "output_dir"    : "./results/reports",
}

# ── Classes du modèle (ordre identique à data.yaml) ──────────
CLASS_NAMES = [
    "bette_sauvage",
    "ble_dur",
    "carotte_sauvage",
    "chardon_marie",
    "chrysantheme",
    "laitue_sauvage",
    "oseille_crepue",
    "oxalis",
    "pisenlit",
    "ray_gras",
]

# ── Paramètres YOLO ───────────────────────────────────────────
YOLO_PARAMS = {
    "conf_threshold": 0.30,
    "iou_threshold" : 0.45,
}

# ── Taux d'occupation réel de la plante dans sa bounding box ─
# 1.0 = la plante remplit 100% de la bbox
# ← Modifie ces valeurs selon tes observations terrain
BBOX_OCCUPATION = {
    "bette_sauvage"  : 0.55,
    "ble_dur"        : 0.40,
    "carotte_sauvage": 0.50,
    "chardon_marie"  : 0.60,
    "chrysantheme"   : 0.65,
    "laitue_sauvage" : 0.70,
    "oseille_crepue" : 0.55,
    "oxalis"         : 0.50,
    "pisenlit"       : 0.60,
    "ray_gras"       : 0.45,
}

# ── Agressivité par espèce ────────────────────────────────────
# 1.0 = neutre | >1.0 = plus agressive | <1.0 = moins agressive
WEED_AGGRESSIVITY = {
    "bette_sauvage"  : 1.2,
    "ble_dur"        : 0.5,   # culture → pas une adventice
    "carotte_sauvage": 1.0,
    "chardon_marie"  : 1.5,
    "chrysantheme"   : 1.3,
    "laitue_sauvage" : 1.1,
    "oseille_crepue" : 1.2,
    "oxalis"         : 1.0,
    "pisenlit"       : 0.9,
    "ray_gras"       : 1.4,
}

# ── Seuils de décision pesticide ─────────────────────────────
PESTICIDE_THRESHOLDS = {
    "red_zone"         : 0.60,  # score > 60% → pulvérisation obligatoire
    "orange_zone"      : 0.30,  # score 30-60% → pulvérisation recommandée
    "yellow_zone"      : 0.05,  # score 5-30%  → surveillance
    "density_critical" : 5,     # nb adventices / m² critique
    "coverage_critical": 0.25,  # 25% surface occupée = critique
}

# ── Doses herbicide de base (L/ha) ───────────────────────────
HERBICIDE_DOSES = {
    "red"   : 3.0,   # zone rouge
    "orange": 1.5,   # zone orange
}

# ── Couleurs de zones (BGR pour OpenCV) ──────────────────────
ZONE_COLORS = {
    "red"   : (0,   0,   220),
    "orange": (0,   140, 255),
    "yellow": (0,   220, 220),
    "green" : (0,   180, 0),
}

ZONE_ALPHA = 0.08  # transparence de la superposition colorée
PATCH_SIZE = 512  # taille d’une cellule de la grille (en pixels)