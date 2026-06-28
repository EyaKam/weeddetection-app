"""
=============================================================
  WEED DETECTION — Lanceur principal du projet
  Détection blé / adventices — Décision pesticides
=============================================================
  UTILISATION :
    python main.py                  → affiche le menu
    python main.py --phase 0        → remapping des classes
    python main.py --phase 1        → split du dataset
    python main.py --phase 2        → vérifie le dataset
    python main.py --phase 3        → évaluation du modèle
    python main.py --phase 4        → inférence + densité
    python main.py --phase 5        → décision pesticides
=============================================================
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────
#  CONFIGURATION GLOBALE DU PROJET
# ──────────────────────────────────────────────
CONFIG = {
    "raw_dataset_original" : "./data/raw",        # export Roboflow brut (36 classes)
    "raw_dataset"          : "./data/remapped",   # après remapping (~10 classes)
    "split_dataset"        : "./data/split",      # résultat du split stratifié

    "model_weights" : "./models/best.pt",
    "base_model"    : "yolov8n.pt",
    "results_dir"   : "./results/reports",

    "train_ratio"   : 0.70,
    "val_ratio"     : 0.20,
    "test_ratio"    : 0.10,
    "random_seed"   : 42,
}


def load_script(name):
    """Charge un script depuis scripts/ par son chemin absolu — évite tout problème sys.path."""
    script_path = Path(__file__).parent / "scripts" / f"{name}.py"
    if not script_path.exists():
        return None
    spec   = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║         WEED DETECTION — Blé & Adventices                ║
║         Système de décision pesticides                   ║
╚══════════════════════════════════════════════════════════╝
    """)


def print_menu():
    print_banner()
    print("  Phases disponibles :\n")
    phases = [
        ("0", "Remapping des classes",     "Fusion 36 → ~10 classes  (raw → remapped)", check_raw_original()),
        ("1", "Préparation des données",   "Split stratifié train/val/test",             check_raw_data()),
        ("2", "Vérification du dataset",   "Statistiques et visualisation des classes",  check_split_data()),
        ("3", "Évaluation du modèle",      "mAP, précision, rappel, confusion matrix",   check_model()),
        ("4", "Inférence + densité",       "Calcul densité et couverture par espèce",    check_model()),
        ("5", "Décision pesticides",       "Recommandation dose par zone",               check_model()),
    ]
    for num, name, desc, ready in phases:
        status = "✅" if ready else "⏳"
        print(f"  {status}  Phase {num} — {name}")
        print(f"           {desc}")
        print()
    print("  Lancement : python main.py --phase <numéro>\n")


# ──────────────────────────────────────────────
#  VÉRIFICATIONS D'ÉTAT
# ──────────────────────────────────────────────

def check_raw_original():
    raw = Path(CONFIG["raw_dataset_original"])
    return (raw / "images").exists() and len(list((raw / "images").glob("*.jpg"))) > 0

def check_raw_data():
    raw = Path(CONFIG["raw_dataset"])
    return (raw / "images").exists() and len(list((raw / "images").glob("*.jpg"))) > 0

def check_split_data():
    return (Path(CONFIG["split_dataset"]) / "train" / "images").exists()

def check_model():
    return Path(CONFIG["model_weights"]).exists()


# ──────────────────────────────────────────────
#  PHASE 0 — REMAPPING DES CLASSES
# ──────────────────────────────────────────────

def run_phase0():
    print("\n━━━  Phase 0 : Remapping des classes  ━━━\n")

    if not check_raw_original():
        print("  ❌ Dataset Roboflow brut introuvable.")
        print(f"     → Images dans   : {CONFIG['raw_dataset_original']}/images/")
        print(f"     → Labels dans   : {CONFIG['raw_dataset_original']}/labels/")
        print(f"     → YAML dans     : {CONFIG['raw_dataset_original']}/data.yaml")
        return

    if check_raw_data():
        confirm = input("  ℹ️  Le dossier remapped existe déjà. Relancer et écraser ? [o/N] : ").strip().lower()
        if confirm != "o":
            print("  ⏭️  Remapping annulé.\n")
            return

    rc = load_script("remap_classes")
    if rc is None:
        print("  ❌ scripts/remap_classes.py introuvable.")
        return

    rc.INPUT_DIR  = CONFIG["raw_dataset_original"]
    rc.OUTPUT_DIR = CONFIG["raw_dataset"]
    rc.main()


# ──────────────────────────────────────────────
#  PHASE 1 — SPLIT STRATIFIÉ
# ──────────────────────────────────────────────

def run_phase1():
    print("\n━━━  Phase 1 : Split stratifié du dataset  ━━━\n")

    if not check_raw_data():
        print("  ❌ Dataset remappé introuvable.")
        print("     → Lance d'abord : python main.py --phase 0")
        return

    ss = load_script("stratified_split")
    if ss is None:
        print("  ❌ scripts/stratified_split.py introuvable.")
        return

    ss.DATASET_DIR = CONFIG["raw_dataset"]
    ss.OUTPUT_DIR  = CONFIG["split_dataset"]
    ss.TRAIN_RATIO = CONFIG["train_ratio"]
    ss.VAL_RATIO   = CONFIG["val_ratio"]
    ss.TEST_RATIO  = CONFIG["test_ratio"]
    ss.RANDOM_SEED = CONFIG["random_seed"]
    ss.main()


# ──────────────────────────────────────────────
#  PHASE 2 — VÉRIFICATION DU DATASET
# ──────────────────────────────────────────────

def run_phase2():
    print("\n━━━  Phase 2 : Vérification du dataset  ━━━\n")

    if not check_split_data():
        print("  ❌ Dataset splitté introuvable.")
        print("     → Lance d'abord : python main.py --phase 1")
        return

    from collections import Counter
    import yaml

    split_dir   = Path(CONFIG["split_dataset"])
    yaml_path   = split_dir / "data.yaml"
    class_names = {}

    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        class_names = {i: n for i, n in enumerate(names)} if isinstance(names, list) else names
    else:
        print("  ⚠️  data.yaml introuvable — classes nommées class_0, class_1 ...")

    print(f"  Classes : {list(class_names.values()) or 'inconnues'}\n")

    total_images = total_instances = 0

    for split in ["train", "val", "test"]:
        lbl_dir = split_dir / split / "labels"
        img_dir = split_dir / split / "images"
        if not lbl_dir.exists():
            continue

        n_images = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        counter  = Counter()
        for lbl in lbl_dir.glob("*.txt"):
            with open(lbl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        counter[int(line.split()[0])] += 1

        n_inst           = sum(counter.values())
        total_images    += n_images
        total_instances += n_inst

        print(f"  [{split.upper():5}]  {n_images:>4} images  |  {n_inst:>5} instances")
        for cid, cnt in sorted(counter.items()):
            name = class_names.get(cid, f"class_{cid}")
            bar  = "█" * int(cnt / max(counter.values()) * 20)
            pct  = cnt / n_inst * 100 if n_inst > 0 else 0
            print(f"           {name:<20} {bar:<20} {cnt:>4} ({pct:.1f}%)")
        print()

    print(f"  TOTAL  {total_images} images  |  {total_instances} instances\n")
    print("  ℹ️  Si une classe dépasse 70% → dataset déséquilibré.")
    print("     Solution : augmenter les données minoritaires dans Roboflow.\n")


# ──────────────────────────────────────────────
#  PHASE 3 — ÉVALUATION DU MODÈLE
# ──────────────────────────────────────────────

def run_phase3():
    print("\n━━━  Phase 3 : Évaluation du modèle  ━━━\n")

    if not check_model():
        print(f"  ❌ Poids introuvables : {CONFIG['model_weights']}")
        print("     → Télécharge best.pt depuis Kaggle → models/")
        return

    if not check_split_data():
        print("  ❌ Dataset de test introuvable → lance d'abord la Phase 1")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ❌ ultralytics non installé → pip install ultralytics")
        return

    data_yaml = str(Path(CONFIG["split_dataset"]) / "data.yaml")
    model     = YOLO(CONFIG["model_weights"])

    print("  Évaluation sur le jeu de test...\n")
    metrics = model.val(data=data_yaml, split="test", verbose=True)

    print(f"\n  ─── Résultats ───")
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Précision : {metrics.box.mp:.4f}")
    print(f"  Rappel    : {metrics.box.mr:.4f}")
    print("\n  Résultats complets → runs/detect/val/\n")


# ──────────────────────────────────────────────
#  PHASE 4 — INFÉRENCE + DENSITÉ & COUVERTURE
# ──────────────────────────────────────────────

def run_phase4():
    print("\n━━━  Phase 4 : Inférence + Densité & Couverture  ━━━\n")

    if not check_model():
        print(f"  ❌ Poids introuvables : {CONFIG['model_weights']}")
        return

    inf = load_script("inference_density")
    if inf is None:
        print("  ⏳ scripts/inference_density.py introuvable.")
        return
    inf.run(model_path=CONFIG["model_weights"], output_dir=CONFIG["results_dir"])


# ──────────────────────────────────────────────
#  PHASE 5 — DÉCISION PESTICIDES
# ──────────────────────────────────────────────

def run_phase5():
    print("\n━━━  Phase 5 : Décision Pesticides  ━━━\n")

    pdec = load_script("pesticide_decision")
    if pdec is None:
        print("  ⏳ scripts/pesticide_decision.py introuvable.")
        return
    pdec.run(results_dir=CONFIG["results_dir"])


# ──────────────────────────────────────────────
#  POINT D'ENTRÉE
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weed Detection — Lanceur principal")
    parser.add_argument(
        "--phase", type=int, choices=[0, 1, 2, 3, 4, 5],
        help="Numéro de la phase à exécuter (0 à 5)"
    )
    args = parser.parse_args()

    if args.phase is None:
        print_menu()
    else:
        {0: run_phase0, 1: run_phase1, 2: run_phase2,
         3: run_phase3, 4: run_phase4, 5: run_phase5}[args.phase]()


if __name__ == "__main__":
    main()