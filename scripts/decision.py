"""
=============================================================
  decision.py — Score d'infection et décision pesticide
=============================================================
"""

from config import (
    WEED_AGGRESSIVITY,
    PESTICIDE_THRESHOLDS,
    HERBICIDE_DOSES,
    ZONE_COLORS,
)


def compute_infection_score(couverture, densite):
    """
    Calcule un score d'infection global pour une cellule (0.0 à 1.0).

    Formule :
      score = 60% × densité pondérée par agressivité
            + 40% × couverture normalisée

    Le blé dur (culture) est exclu du calcul.
    """
    if not couverture:
        return 0.0

    # Densité pondérée par agressivité (hors blé)
    density_score = sum(
        pct * WEED_AGGRESSIVITY.get(name, 1.0)
        for name, pct in densite.items()
        if name != "ble_dur"
    )

    # Couverture normalisée (hors blé)
    weed_count  = sum(v for k, v in couverture.items() if k != "ble_dur")
    count_score = min(weed_count / PESTICIDE_THRESHOLDS["density_critical"], 1.0)

    score = 0.60 * min(density_score, 1.0) + 0.40 * count_score
    return round(min(score, 1.0), 4)


def get_zone(infection_score):
    """
    Retourne le label de zone, la couleur BGR et la description
    selon le score d'infection.

    Zones :
      red    → score ≥ 0.60 → pulvérisation obligatoire
      orange → score ≥ 0.30 → pulvérisation recommandée
      yellow → score ≥ 0.05 → surveillance
      green  → score <  0.05 → sain
    """
    if infection_score >= PESTICIDE_THRESHOLDS["red_zone"]:
        return "red",    ZONE_COLORS["red"],    "🔴 Pulvérisation obligatoire"
    elif infection_score >= PESTICIDE_THRESHOLDS["orange_zone"]:
        return "orange", ZONE_COLORS["orange"], "🟠 Pulvérisation recommandée"
    elif infection_score >= PESTICIDE_THRESHOLDS["yellow_zone"]:
        return "yellow", ZONE_COLORS["yellow"], "🟡 Surveillance"
    else:
        return "green",  ZONE_COLORS["green"],  "🟢 Sain"


def decide_pesticide(couverture, densite, infection_score):
    """
    Décision de pulvérisation basée sur le score d'infection.

    Retourne un dict :
    {
      "zone"              : str,
      "description"       : str,
      "infection_score"   : float,
      "weed_count"        : int,
      "total_coverage_pct": float,
      "pulverisation"     : bool,
      "herbicide_dose"    : str | None,
      "especes_cibles"    : [str],
    }
    """
    zone_label, _, zone_desc = get_zone(infection_score)

    weed_count     = sum(v for k, v in couverture.items() if k != "ble_dur")
    total_coverage = sum(v for k, v in densite.items()    if k != "ble_dur")
    especes        = [k for k in couverture if k != "ble_dur"]

    pulverisation = zone_label in ("red", "orange")

    herbicide_dose = None
    if pulverisation:
        base_dose = HERBICIDE_DOSES.get(zone_label, 1.5)
        agr_max   = max((WEED_AGGRESSIVITY.get(e, 1.0) for e in especes),
                        default=1.0)
        herbicide_dose = f"{round(base_dose * agr_max, 2)} L/ha"

    return {
        "zone"              : zone_label,
        "description"       : zone_desc,
        "infection_score"   : infection_score,
        "weed_count"        : weed_count,
        "total_coverage_pct": round(total_coverage * 100, 2),
        "pulverisation"     : pulverisation,
        "herbicide_dose"    : herbicide_dose,
        "especes_cibles"    : especes,
    }
