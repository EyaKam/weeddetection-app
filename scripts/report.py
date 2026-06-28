"""
=============================================================
  report.py — Génération des rapports JSON et TXT
=============================================================
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def build_report_data(image_path, cells, cell_results,
                      n_rows, n_cols, pixels_per_meter):
    """
    Construit le dictionnaire complet du rapport
    (structure prête à être sérialisée en JSON).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "image"            : str(image_path),
        "timestamp"        : timestamp,
        "pixels_per_meter" : round(pixels_per_meter, 2),
        "grid"             : {
            "rows"        : n_rows,
            "cols"        : n_cols,
            "total_cells" : len(cells),
        },
        "cells"   : [],
        "summary" : {},
    }

    zone_counts   = defaultdict(int)
    all_especes   = defaultdict(int)
    pulv_cells    = []

    for cell in cells:
        r, c = cell["row"], cell["col"]
        res  = cell_results.get((r, c), {})
        dec  = res.get("decision", {})

        cell_entry = {
            "id"               : f"R{r+1}C{c+1}",
            "row"              : r + 1,
            "col"              : c + 1,
            "coords_px"        : {
                "x1": cell["x1"], "y1": cell["y1"],
                "x2": cell["x2"], "y2": cell["y2"],
            },
            "infection_score"  : res.get("infection_score", 0),
            "zone"             : dec.get("zone", "green"),
            "couverture"       : res.get("couverture", {}),
            "densite_pct"      : {
                k: round(v * 100, 2)
                for k, v in res.get("densite", {}).items()
            },
            "decision"         : dec,
            "detections_count" : len(res.get("detections", [])),
        }

        report["cells"].append(cell_entry)

        zone = cell_entry["zone"]
        zone_counts[zone] += 1

        for esp, cnt in res.get("couverture", {}).items():
            if esp != "ble_dur":
                all_especes[esp] += cnt

        if dec.get("pulverisation"):
            pulv_cells.append(cell_entry["id"])

    total = len(cells)
    report["summary"] = {
        "total_m2"           : total,
        "zones"              : dict(zone_counts),
        "pct_red"            : round(zone_counts["red"]    / total * 100, 1),
        "pct_orange"         : round(zone_counts["orange"] / total * 100, 1),
        "pct_yellow"         : round(zone_counts["yellow"] / total * 100, 1),
        "pct_green"          : round(zone_counts["green"]  / total * 100, 1),
        "especes_detectees"  : dict(sorted(all_especes.items(),
                                           key=lambda x: x[1], reverse=True)),
        "zones_pulverisation": pulv_cells,
        "nb_m2_a_pulveriser" : len(pulv_cells),
    }

    return report, timestamp


def save_json(report, output_dir, img_name, timestamp):
    """Sauvegarde le rapport au format JSON."""
    path = Path(output_dir) / f"{img_name}_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def save_txt(report, output_dir, img_name, timestamp):
    """Sauvegarde le rapport au format texte lisible."""
    path = Path(output_dir) / f"{img_name}_{timestamp}.txt"
    s    = report["summary"]
    g    = report["grid"]

    with open(path, "w", encoding="utf-8") as f:

        # En-tête
        f.write("=" * 65 + "\n")
        f.write("  RAPPORT D'ANALYSE — WEED DETECTION\n")
        f.write(f"  Image  : {report['image']}\n")
        f.write(f"  Date   : {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
        f.write(f"  Grille : {g['rows']} x {g['cols']} = {g['total_cells']} m2\n")
        f.write("=" * 65 + "\n\n")

        # Résumé global
        f.write("-- RESUME GLOBAL ------------------------------------------\n")
        f.write(f"  Surface totale analysee : {s['total_m2']} m2\n")
        f.write(f"  Zones rouges   (tres infectees)  : {s['zones'].get('red',0):>4} m2"
                f" ({s['pct_red']}%)\n")
        f.write(f"  Zones oranges  (mod. infectees)  : {s['zones'].get('orange',0):>4} m2"
                f" ({s['pct_orange']}%)\n")
        f.write(f"  Zones jaunes   (peu infectees)   : {s['zones'].get('yellow',0):>4} m2"
                f" ({s['pct_yellow']}%)\n")
        f.write(f"  Zones vertes   (saines)          : {s['zones'].get('green',0):>4} m2"
                f" ({s['pct_green']}%)\n")
        f.write(f"\n  Surface a pulveriser : {s['nb_m2_a_pulveriser']} m2\n")

        if s["zones_pulverisation"]:
            f.write(f"  Zones concernees     : {', '.join(s['zones_pulverisation'])}\n")

        f.write("\n  Especes detectees (adventices) :\n")
        for esp, cnt in s["especes_detectees"].items():
            f.write(f"    {esp:<22} : {cnt} individus\n")

        # Détail par cellule
        f.write("\n\n-- DETAIL PAR METRE CARRE ---------------------------------\n")
        for cell in report["cells"]:
            if cell["detections_count"] == 0:
                continue

            f.write(f"\n  [{cell['id']}]  Zone : {cell['zone'].upper()}"
                    f"  |  Score : {cell['infection_score']:.3f}\n")

            if cell["couverture"]:
                f.write("    Couverture (individus) :\n")
                for esp, cnt in cell["couverture"].items():
                    f.write(f"      {esp:<22} : {cnt} plants\n")

            if cell["densite_pct"]:
                f.write("    Densite (surface occupee) :\n")
                for esp, pct in cell["densite_pct"].items():
                    bar = ">" * int(pct / 5)
                    f.write(f"      {esp:<22} : {bar:<20} {pct:.1f}%\n")

            dec = cell["decision"]
            if dec.get("pulverisation"):
                f.write(f"    PULVERISATION : {dec.get('herbicide_dose', 'N/A')}\n")
                f.write(f"    Especes cibles : {', '.join(dec.get('especes_cibles', []))}\n")

        f.write("\n" + "=" * 65 + "\n")
        f.write("  Fin du rapport\n")
        f.write("=" * 65 + "\n")

    return path


def generate_report(image_path, cells, cell_results,
                    n_rows, n_cols, pixels_per_meter, output_dir):
    """
    Point d'entrée : génère et sauvegarde JSON + TXT.
    Retourne les chemins des deux fichiers.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    img_name = Path(image_path).stem

    report, timestamp = build_report_data(
        image_path, cells, cell_results,
        n_rows, n_cols, pixels_per_meter
    )

    json_path = save_json(report, output_dir, img_name, timestamp)
    txt_path  = save_txt(report, output_dir, img_name, timestamp)

    return json_path, txt_path, report
