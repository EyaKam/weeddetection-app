"""
=============================================================
  inference.py — Modèle 1 (détection espèces)
               + Modèle 2 (classification stade)
=============================================================
"""

import sys
from pathlib import Path

import cv2

from config import CLASS_NAMES, YOLO_PARAMS, PATHS


# ── Mapping classe → stades possibles ────────────────────────
# Utilisé pour filtrer les prédictions du Modèle 2
CLASS_STAGES = {
    "bette_sauvage"  : ["bette_sauvage_s1", "bette_sauvage_s2"],
    "ble_dur"        : ["ble_dur_s1",        "ble_dur_s2",       "ble_dur_s3"],
    "carotte_sauvage": ["carotte_sauvage_s1","carotte_sauvage_s2"],
    "chardon_marie"  : ["chardon_marie_s1",  "chardon_marie_s2",
                        "chardon_marie_s3",  "chardon_marie_s4"],
    "chrysantheme"   : ["chrysantheme_s1",   "chrysantheme_s2",
                        "chrysantheme_s3",   "chrysantheme_s4"],
    "laitue_sauvage" : ["laitue_sauvage_s1", "laitue_sauvage_s2",
                        "laitue_sauvage_s3", "laitue_sauvage_s4"],
    "oseille_crepue" : ["oseille_crepue_s1", "oseille_crepue_s2"],
    "oxalis"         : ["oxalis_s1",         "oxalis_s2",
                        "oxalis_s3",         "oxalis_s4"],
    "pisenlit"       : ["pisenlit_s1",       "pisenlit_s2",      "pisenlit_s3"],
    "ray_gras"       : ["ray_gras_s1",       "ray_gras_s2",
                        "ray_gras_s3"],
}


def load_model(model_path, task="detect"):
    """
    Charge un modèle YOLO.
    task = 'detect'   → Modèle 1 (détection espèces)
    task = 'classify' → Modèle 2 (classification stades)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ❌ ultralytics non installé → pip install ultralytics")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"  ❌ Modèle introuvable : {model_path}")
        sys.exit(1)

    model = YOLO(model_path)
    label = "Détection" if task == "detect" else "Classification"
    print(f"  ✅ Modèle {label} chargé : {model_path}")
    return model


def load_both_models():
    """Charge Modèle 1 (espèces) et Modèle 2 (stades)."""
    print("\n  Chargement des modèles...")
    model1 = load_model(PATHS["model_global"],  task="detect")
    model2 = load_model(PATHS["model_stages"],  task="classify")
    return model1, model2


def classify_stage(model2, crop_img, class_name):
    """
    Modèle 2 — classifie le stade d'un crop.

    Filtre les prédictions pour ne garder que les stades
    compatibles avec l'espèce détectée par Modèle 1.

    Retourne : (stage_label, confidence)
    ex: ("s2", 0.87)
    """
    if crop_img is None or crop_img.size == 0:
        return "inconnu", 0.0

    results = model2(crop_img, verbose=False)
    probs   = results[0].probs
    names   = results[0].names

    # Stades valides pour cette espèce
    valid_stages = CLASS_STAGES.get(class_name, [])

    if not valid_stages:
        # Si pas de mapping → retourne la top prédiction brute
        top1_name = names[probs.top1]
        return top1_name, float(probs.top1conf)

    # Cherche la meilleure prédiction parmi les stades valides
    best_stage = "inconnu"
    best_conf  = 0.0

    for i, name in names.items():
        if name in valid_stages:
            conf = float(probs.data[i])
            if conf > best_conf:
                best_conf  = conf
                best_stage = name

    # Extrait juste le suffixe de stade (ex: "chrysantheme_s2" → "s2")
    if "_s" in best_stage:
        stage_label = "s" + best_stage.split("_s")[-1]
    else:
        stage_label = best_stage

    return stage_label, round(best_conf, 4)


def predict_cell(model1, model2, cell_img):
    """
    Pipeline complet pour une cellule 1m² :
      1. Modèle 1 → détecte espèces + bboxes
      2. Pour chaque détection → crop → Modèle 2 → stade

    Retourne une liste de détections enrichies :
    [
      {
        "class_name" : "chrysantheme",
        "stage"      : "s2",
        "confidence" : 0.87,
        "stage_conf" : 0.79,
        "x1", "y1", "x2", "y2",
        "bbox_area_px",
      },
      ...
    ]
    """
    # ── Modèle 1 : détection ──
    results = model1(
        cell_img,
        conf    = YOLO_PARAMS["conf_threshold"],
        iou     = YOLO_PARAMS["iou_threshold"],
        verbose = False,
    )

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = (CLASS_NAMES[class_id]
                          if class_id < len(CLASS_NAMES)
                          else f"class_{class_id}")

            # ── Modèle 2 : stade ──
            ix1 = max(0, int(x1))
            iy1 = max(0, int(y1))
            ix2 = min(cell_img.shape[1], int(x2))
            iy2 = min(cell_img.shape[0], int(y2))
            crop = cell_img[iy1:iy2, ix1:ix2]
            crop_resized = cv2.resize(crop, (224, 224)) if crop.size > 0 else None

            stage, stage_conf = classify_stage(model2, crop_resized, class_name)

            detections.append({
                "class_id"    : class_id,
                "class_name"  : class_name,
                "stage"       : stage,
                "confidence"  : round(confidence, 4),
                "stage_conf"  : stage_conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "bbox_area_px": (x2 - x1) * (y2 - y1),
            })

    return detections


def run_inference_on_all_cells(model1, model2, cells):
    """
    Lance le pipeline Modèle1 + Modèle2 sur toutes les cellules.

    Retourne : {(row, col): [detections_enrichies]}
    """
    print(f"\n  🔍 Inférence sur {len(cells)} cellules...")
    all_detections = {}
    total = len(cells)

    for i, cell in enumerate(cells):
        key = (cell["row"], cell["col"])
        all_detections[key] = predict_cell(model1, model2, cell["image"])
        n = len(all_detections[key])
        print(f"  [{i+1:>4}/{total}] R{cell['row']+1}C{cell['col']+1}"
              f" — {n} détections", end="\r")

    print(f"\n  ✅ Inférence terminée — pipeline Modèle1 + Modèle2")
    return all_detections