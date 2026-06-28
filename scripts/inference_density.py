"""
=============================================================
  inference_density.py — Orchestrateur principal Phase 4
  Weed Detection : Inférence + Densité + Couverture + Rapport
=============================================================
  UTILISATION :
    python scripts/inference_density.py --image <chemin_image>

  EXEMPLE :
    python scripts/inference_density.py --image ./results/test_composite.jpg
=============================================================
"""

import argparse
import sys
import cv2
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config            import PATHS
from cell_grid         import run_grid
from inference_updated import load_both_models, run_inference_on_all_cells
from features          import compute_features
from decision          import compute_infection_score, decide_pesticide
from visualization     import build_output_image, save_outputs
from report            import generate_report
from datetime import datetime

def analyse_image(image_path, model_path=None, output_dir=None):
    """
    Pipeline complet :
      1. Chargement image + Modèle 1 (espèces) + Modèle 2 (stades)
      2. Division en cellules (mode patch ou référence manuelle)
      3. Inférence Modèle 1 → détection espèces
         Inférence Modèle 2 → classification stade par crop
      4. Calcul couverture + densité par cellule
      5. Score d'infection + décision pesticide
      6. Génération image annotée
      7. Génération rapport JSON + TXT
      8. Résumé console
    """
    model_path = model_path or PATHS["model_global"]
    output_dir = output_dir or PATHS["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Chargement ─────────────────────────────────────────
    print(f"\n🌾  Analyse : {image_path}")
    print("─" * 55)
    print("\n  Étape 1/7 — Chargement image + modèles")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ❌ Image introuvable : {image_path}")
        return

    h, w = image.shape[:2]
    print(f"  Image chargée : {w} × {h} px")

    # Charge Modèle 1 (espèces) + Modèle 2 (stades)
    model1, model2 = load_both_models()

    # ── 2. Division en cellules ───────────────────────────────
    print("\n  Étape 2/7 — Division en cellules")
    cells, n_rows, n_cols, pixels_per_meter = run_grid(image)

    # ── 3. Inférence ──────────────────────────────────────────
    print("\n  Étape 3/7 — Inférence Modèle 1 + Modèle 2")
    all_detections = run_inference_on_all_cells(model1, model2, cells)

    # ── 4. Features + Décision ───────────────────────────────
    print("\n  Étape 4/7 — Calcul couverture + densité")
    cell_results = {}

    for cell in cells:
        key        = (cell["row"], cell["col"])
        detections = all_detections.get(key, [])

        couverture, densite = compute_features(detections, cell["area_px"])
        infection_score     = compute_infection_score(couverture, densite)
        decision            = decide_pesticide(couverture, densite, infection_score)

        cell_results[key] = {
            "detections"     : detections,
            "couverture"     : couverture,
            "densite"        : densite,
            "infection_score": infection_score,
            "decision"       : decision,
        }

    print(f"  ✅ {len(cells)} cellules traitées")

    # ── 5. Image annotée ─────────────────────────────────────
    print("\n  Étape 5/7 — Génération des 3 images")

    img_name  = Path(image_path).stem                        # ← ajoute
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    # ← ajoute

    img_bboxes, img_zones, img_combined = build_output_image(
        image, cells, cell_results
    )
    save_outputs(img_bboxes, img_zones, img_combined,
                output_dir, img_name, timestamp)
    # ── 6. Rapport ───────────────────────────────────────────
    print("\n  Étape 6/7 — Génération rapport")
    json_path, txt_path, report = generate_report(
        image_path, cells, cell_results,
        n_rows, n_cols, pixels_per_meter, output_dir
    )
    print(f"  ✅ Rapport JSON : {json_path}")
    print(f"  ✅ Rapport TXT  : {txt_path}")

    # ── 7. Résumé console ────────────────────────────────────
    print("\n  Étape 7/7 — Résumé")
    s = report["summary"]

    print(f"""
  ══════════════════════════════════════════
    RÉSUMÉ — {s['total_m2']} cellules analysées
  ══════════════════════════════════════════
    🔴 Très infectées       : {s['zones'].get('red',    0):>4}  ({s['pct_red']}%)
    🟠 Modérément infectées : {s['zones'].get('orange', 0):>4}  ({s['pct_orange']}%)
    🟡 Peu infectées        : {s['zones'].get('yellow', 0):>4}  ({s['pct_yellow']}%)
    🟢 Saines               : {s['zones'].get('green',  0):>4}  ({s['pct_green']}%)
  ──────────────────────────────────────────
    ⚠️  À pulvériser        : {s['nb_m2_a_pulveriser']:>4} cellules
  ══════════════════════════════════════════""")

    if s["especes_detectees"]:
        print("\n  Espèces détectées :")
        for esp, cnt in list(s["especes_detectees"].items())[:5]:
            print(f"    • {esp:<22} : {cnt} individus")

    print(f"\n  📁 Résultats dans  : {output_dir}")
    print(f"  🖼️  Ouvre les images dans : {output_dir}\n")


# ══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Weed Detection — Pipeline complet"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Chemin vers l'image à analyser"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Chemin vers best.pt Modèle 1 (défaut : models/best.pt)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Dossier de sortie (défaut : results/reports)"
    )
    args = parser.parse_args()

    analyse_image(
        image_path = args.image,
        model_path = args.model,
        output_dir = args.output,
    )


if __name__ == "__main__":
    main()