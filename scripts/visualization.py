"""
=============================================================
  visualization.py — Génération des images de sortie

  Génère 3 images séparées :
    ① _bboxes.jpg   → bboxes uniquement (style YOLO)
    ② _zones.jpg    → zones d'infection uniquement
    ③ _combined.jpg → bboxes + zones ensemble

  Paramètres utilisateur conservés :
    GREY_FACTOR    = 0.75
    ZONE_ALPHA     = 0.55
    BBOX_THICKNESS = 4
    FONT_THICKNESS = 4

  Couleurs : palette terreuse et naturelle (pas fluorescente)
=============================================================
"""

import cv2
import numpy as np
from pathlib import Path
from decision import get_zone

# ── Palette terreuse — couleurs naturelles de terrain (BGR) ──
CLASS_COLORS = {
    "bette_sauvage"  : ( 80, 100, 140),   # brun-gris
    "ble_dur"        : ( 60, 120,  80),   # vert forêt
    "carotte_sauvage": ( 50,  80, 160),   # rouille doux
    "chardon_marie"  : (100, 120,  60),   # olive sombre
    "chrysantheme"   : (100,  70, 120),   # prune doux
    "laitue_sauvage" : ( 60, 130, 110),   # vert-sauge
    "oseille_crepue" : ( 50, 100, 150),   # terracotta doux
    "oxalis"         : (110,  70, 100),   # rose poudré
    "pisenlit"       : ( 50, 130, 120),   # moutarde terreuse
    "ray_gras"       : ( 70, 130,  70),   # vert mousse
}
DEFAULT_COLOR = (100, 100, 100)

# ── Couleurs zones — terreuses et sombres (BGR) ───────────────
ZONE_COLORS = {
    "red"   : ( 30,  30, 160),   # rouge brique sombre
    "orange": ( 20,  90, 170),   # orange terre sombre
    "yellow": ( 30, 140, 140),   # ocre doux
    "green" : ( 30,  90,  30),   # vert forêt sombre
}

# ── Paramètres (conservés depuis ta version) ─────────────────
GREY_FACTOR    = 0.75
ZONE_ALPHA     = 0.55
BBOX_THICKNESS = 8
FONT_THICKNESS = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX


# ══════════════════════════════════════════════════════════════
#  COUCHE DE BASE
# ══════════════════════════════════════════════════════════════

def grey_image(image, factor=GREY_FACTOR):
    return np.clip(image.astype(np.float32) * factor,
                   0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════
#  ZONES D'INFECTION
# ══════════════════════════════════════════════════════════════

def draw_infection_zones(image, cells, cell_results):
    """
    Colore TOUTE la surface de chaque cellule infectée.
    Flou gaussien fort → effet heatmap naturel sans boîtes rigides.
    """
    h, w    = image.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    for cell in cells:
        key    = (cell["row"], cell["col"])
        result = cell_results.get(key, {})
        score  = result.get("infection_score", 0.0)

        zone_label, _, _ = get_zone(score)
        if zone_label == "green":
            continue

        color = ZONE_COLORS[zone_label]
        cv2.rectangle(overlay,
                      (cell["x1"], cell["y1"]),
                      (cell["x2"], cell["y2"]),
                      color, -1)

    # Flou fort → frontières entre cellules disparaissent
    overlay_blurred = cv2.GaussianBlur(overlay, (151, 151), 0)

    output = image.copy()
    mask   = overlay_blurred.sum(axis=2) > 0
    mask_3 = np.stack([mask, mask, mask], axis=2)
    blended = cv2.addWeighted(overlay_blurred, ZONE_ALPHA,
                              output, 1 - ZONE_ALPHA, 0)
    output[mask_3] = blended[mask_3]

    return output


# ══════════════════════════════════════════════════════════════
#  BBOXES + LABELS
# ══════════════════════════════════════════════════════════════

def draw_detections(image, cells, cell_results):
    """
    Dessine les bboxes et labels de chaque détection.
    Couleurs terreuses, fond semi-transparent sur le label.
    """
    output       = image.copy()
    img_h, img_w = output.shape[:2]
    font_scale   = max(0.38, min(0.55, img_w / 3000))

    for cell in cells:
        key  = (cell["row"], cell["col"])
        dets = cell_results.get(key, {}).get("detections", [])

        for det in dets:
            abs_x1 = cell["x1"] + int(det["x1"])
            abs_y1 = cell["y1"] + int(det["y1"])
            abs_x2 = cell["x1"] + int(det["x2"])
            abs_y2 = cell["y1"] + int(det["y2"])

            name  = det.get("class_name", "?")
            stage = det.get("stage", "")
            conf  = det.get("confidence", 0)
            color = CLASS_COLORS.get(name)

            # Bbox
            cv2.rectangle(output,
                          (abs_x1, abs_y1), (abs_x2, abs_y2),
                          color, BBOX_THICKNESS)

            # Label
            label = f"{name}"
            if stage and stage not in ("inconnu", ""):
                label += f" {stage}"
            label += f" {conf:.2f}"

            (tw, th), _ = cv2.getTextSize(
                label, FONT, font_scale, FONT_THICKNESS)

            lx1 = max(0, abs_x1)
            ly1 = max(0, abs_y1 - th - 4)
            lx2 = min(img_w, abs_x1 + tw + 6)
            ly2 = max(ly1 + 1, min(img_h, abs_y1))

            # Fond semi-transparent
            if lx2 > lx1 and ly2 > ly1:
                sub = output[ly1:ly2, lx1:lx2].copy()
                if sub.size > 0:
                    bg = np.full_like(sub, color)
                    output[ly1:ly2, lx1:lx2] = cv2.addWeighted(
                        bg, 0.75, sub, 0.25, 0)

            # Texte blanc
            text_y = max(abs_y1 - 4, th + 4)
            cv2.putText(output, label,
                        (abs_x1 + 3, text_y),
                        FONT, font_scale,
                        (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)

    return output


# ══════════════════════════════════════════════════════════════
#  LÉGENDE
# ══════════════════════════════════════════════════════════════

def add_legend(image, cell_results, show_zones=True, show_classes=True):
    h, w     = image.shape[:2]
    legend_h = 90
    legend   = np.full((legend_h, w, 3), 20, dtype=np.uint8)

    x = 15
    # ── Zones ──
    if show_zones:
        cv2.putText(legend, "ZONES:", (x, 28),
                    FONT, 0.48, (180, 180, 180), 1)
        x += 80
        for label, color in [
            ("Tres infecte",   ZONE_COLORS["red"]),
            ("Infecte",        ZONE_COLORS["orange"]),
            ("Peu infecte",    ZONE_COLORS["yellow"]),
            ("Sain",           ZONE_COLORS["green"]),
        ]:
            cv2.rectangle(legend, (x, 12), (x + 22, 34), color, -1)
            cv2.putText(legend, label, (x + 28, 28),
                        FONT, 0.40, (210, 210, 210), 1)
            x += 145

    # ── Classes ──
    if show_classes:
        detected = sorted(set(
            det.get("class_name", "?")
            for res in cell_results.values()
            for det in res.get("detections", [])
        ))

        x = 15
        cv2.putText(legend, "CLASSES:", (x, 68),
                    FONT, 0.48, (180, 180, 180), 1)
        x += 95
        step = max(110, (w - 100) // max(len(detected), 1))
        for cls in detected:
            color = CLASS_COLORS.get(cls)
            cv2.rectangle(legend, (x, 54), (x + 16, 74), color, -1)
            short = cls.replace("_", " ")
            cv2.putText(legend, short, (x + 20, 69),
                        FONT, 0.36, (210, 210, 210), 1)
            x += step
            if x > w - 120:
                break

    return np.vstack([image, legend])


# ══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE — génère 3 images
# ══════════════════════════════════════════════════════════════

def build_output_image(image, cells, cell_results):
    """
    Génère 3 images depuis la même base :
      img_bboxes   → bboxes uniquement
      img_zones    → zones d'infection uniquement
      img_combined → bboxes + zones ensemble
    """
    base = grey_image(image)

    # ① Bboxes uniquement
    img_bboxes = draw_detections(base.copy(), cells, cell_results)
    img_bboxes = add_legend(img_bboxes, cell_results,
                            show_zones=False, show_classes=True)

    # ② Zones uniquement
    img_zones = draw_infection_zones(base.copy(), cells, cell_results)
    img_zones = add_legend(img_zones, cell_results,
                           show_zones=True, show_classes=False)

    # ③ Combined — zones d'abord puis bboxes par-dessus
    img_combined = draw_infection_zones(base.copy(), cells, cell_results)
    img_combined = draw_detections(img_combined, cells, cell_results)
    img_combined = add_legend(img_combined, cell_results,
                              show_zones=True, show_classes=True)

    return img_bboxes, img_zones, img_combined


def save_outputs(img_bboxes, img_zones, img_combined,
                 output_dir, img_name, timestamp):
    """Sauvegarde les 3 images et retourne leurs chemins."""
    out  = Path(output_dir)
    opts = [cv2.IMWRITE_JPEG_QUALITY, 95]

    paths = {
        "bboxes"  : out / f"{img_name}_bboxes_{timestamp}.jpg",
        "zones"   : out / f"{img_name}_zones_{timestamp}.jpg",
        "combined": out / f"{img_name}_combined_{timestamp}.jpg",
    }

    cv2.imwrite(str(paths["bboxes"]),   img_bboxes,   opts)
    cv2.imwrite(str(paths["zones"]),    img_zones,    opts)
    cv2.imwrite(str(paths["combined"]), img_combined, opts)

    print(f"  ✅ Bboxes   : {paths['bboxes']}")
    print(f"  ✅ Zones    : {paths['zones']}")
    print(f"  ✅ Combined : {paths['combined']}")

    return paths