"""
=============================================================
  features.py — Calcul des features par cellule
                 • Couverture : nb individus par espèce
                 • Densité    : surface occupée (%) par espèce
=============================================================
"""

from collections import defaultdict
from config import BBOX_OCCUPATION


def compute_couverture(detections):
    """
    Couverture = nombre d'individus par espèce dans la cellule.

    Retourne : {"chrysantheme": 3, "oxalis": 5, ...}
    """
    couverture = defaultdict(int)
    for det in detections:
        couverture[det["class_name"]] += 1
    return dict(couverture)


def compute_densite(detections, cell_area_px):
    """
    Densité = surface réelle occupée par chaque espèce / surface cellule.

    Surface réelle = surface bbox × taux d'occupation (BBOX_OCCUPATION).
    Résultat exprimé en proportion (0.0 à 1.0) → 1.0 = 100% de la cellule.

    Retourne : {"chrysantheme": 0.12, "oxalis": 0.08, ...}
    """
    surface_px = defaultdict(float)

    for det in detections:
        name      = det["class_name"]
        occ_ratio = BBOX_OCCUPATION.get(name, 0.50)
        surface_px[name] += det["bbox_area_px"] * occ_ratio

    densite = {
        name: min(surf / cell_area_px, 1.0)
        for name, surf in surface_px.items()
    }
    return densite


def compute_features(detections, cell_area_px):
    """
    Point d'entrée : calcule couverture ET densité pour une cellule.

    Retourne :
      couverture : dict {espèce: nb_individus}
      densite    : dict {espèce: proportion_surface (0-1)}
    """
    couverture = compute_couverture(detections)
    densite    = compute_densite(detections, cell_area_px)
    return couverture, densite
