"""
=============================================================
  Stratified Train/Val/Test Split — Format YOLO
  Projet : Détection blé / adventices
=============================================================
  UTILISATION :
    1. Exporte ton dataset Roboflow en format YOLOv8
    2. Modifie le chemin DATASET_DIR ci-dessous
    3. Lance :  python stratified_split.py
=============================================================
"""

import os
import shutil
import random
from collections import defaultdict, Counter
from pathlib import Path
import yaml

# ──────────────────────────────────────────────
#  CONFIGURATION  ← modifie ici
# ──────────────────────────────────────────────
DATASET_DIR  = "./dataset"          # dossier racine exporté depuis Roboflow
OUTPUT_DIR   = "./dataset_split"    # où les splits seront créés
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.20
TEST_RATIO   = 0.10
RANDOM_SEED  = 42                   # pour la reproductibilité
# ──────────────────────────────────────────────

random.seed(RANDOM_SEED)

def load_dataset(dataset_dir):
    """Charge toutes les images et leurs annotations YOLO."""
    images_dir = Path(dataset_dir) / "images"
    labels_dir = Path(dataset_dir) / "labels"

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    samples = []

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue

        label_path = labels_dir / (img_path.stem + ".txt")

        # Classes présentes dans cette image
        classes_in_image = set()
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        classes_in_image.add(class_id)

        samples.append({
            "image": img_path,
            "label": label_path if label_path.exists() else None,
            "classes": classes_in_image
        })

    return samples


def stratified_split(samples, train_r, val_r, test_r):
    """
    Stratified split basé sur la classe DOMINANTE de chaque image.
    Les images multi-classes sont réparties pour équilibrer toutes les classes.
    """
    # Grouper par classe dominante (classe la plus fréquente dans l'image)
    class_to_samples = defaultdict(list)

    for s in samples:
        if s["classes"]:
            # Utilise la première classe comme clé de stratification
            # (on peut améliorer avec la classe dominante par comptage)
            dominant = sorted(s["classes"])[0]
        else:
            dominant = -1  # images sans annotation
        class_to_samples[dominant].append(s)

    train, val, test = [], [], []

    for cls, cls_samples in class_to_samples.items():
        random.shuffle(cls_samples)
        n = len(cls_samples)
        n_train = max(1, round(n * train_r))
        n_val   = max(1, round(n * val_r))
        n_test  = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val  = n - n_train

        train += cls_samples[:n_train]
        val   += cls_samples[n_train:n_train + n_val]
        test  += cls_samples[n_train + n_val:]

    return train, val, test


def copy_split(samples, split_name, output_dir):
    """Copie les fichiers image + label dans le dossier de split."""
    img_out = Path(output_dir) / split_name / "images"
    lbl_out = Path(output_dir) / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for s in samples:
        shutil.copy2(s["image"], img_out / s["image"].name)
        if s["label"] and s["label"].exists():
            shutil.copy2(s["label"], lbl_out / s["label"].name)

    return len(samples)


def load_class_names(dataset_dir):
    """Lit les noms de classes depuis data.yaml."""
    yaml_path = Path(dataset_dir) / "data.yaml"
    if not yaml_path.exists():
        return {}
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return names


def count_instances(samples, class_names):
    """Compte les instances par classe dans un split."""
    counter = Counter()
    for s in samples:
        if s["label"] and s["label"].exists():
            with open(s["label"]) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        counter[int(line.split()[0])] += 1
    result = {}
    for cid, count in sorted(counter.items()):
        name = class_names.get(cid, f"class_{cid}")
        result[name] = count
    return result


def print_distribution(train, val, test, class_names):
    """Affiche un tableau de distribution propre."""
    splits = {"TRAIN": train, "VAL": val, "TEST": test}
    all_counts = {k: count_instances(v, class_names) for k, v in splits.items()}

    # Toutes les classes présentes
    all_classes = sorted(set(
        cls for counts in all_counts.values() for cls in counts
    ))

    col_w = 14
    header = f"{'Classe':<20}" + "".join(f"{s:>{col_w}}" for s in ["TRAIN", "VAL", "TEST", "TOTAL"])
    print("\n" + "=" * len(header))
    print("  DISTRIBUTION DES INSTANCES PAR CLASSE ET PAR SPLIT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    totals = {s: 0 for s in splits}
    for cls in all_classes:
        row = f"{cls:<20}"
        total = 0
        for split_name in ["TRAIN", "VAL", "TEST"]:
            count = all_counts[split_name].get(cls, 0)
            totals[split_name] += count
            total += count
            row += f"{count:>{col_w}}"
        row += f"{total:>{col_w}}"
        print(row)

    print("-" * len(header))
    total_all = sum(totals.values())
    totals_row = f"{'TOTAL':<20}" + "".join(f"{totals[s]:>{col_w}}" for s in ["TRAIN", "VAL", "TEST"])
    totals_row += f"{total_all:>{col_w}}"
    print(totals_row)
    print("=" * len(header))

    # Ratios réels
    print("\n  RATIOS RÉELS (images)")
    n_total = len(train) + len(val) + len(test)
    for name, split in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        pct = len(split) / n_total * 100 if n_total > 0 else 0
        print(f"    {name:<8} : {len(split):>4} images  ({pct:.1f}%)")
    print(f"    {'TOTAL':<8} : {n_total:>4} images")


def create_yaml(output_dir, dataset_dir, class_names):
    """Crée le data.yaml pour l'entraînement YOLO."""
    yaml_content = {
        "path": str(Path(output_dir).resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(class_names),
        "names": [class_names[i] for i in sorted(class_names)]
    }
    out_yaml = Path(output_dir) / "data.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    print(f"\n  data.yaml créé → {out_yaml}")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main():
    print(f"\n🌾  Stratified Split — Détection blé/adventices")
    print(f"   Dataset source : {DATASET_DIR}")
    print(f"   Output         : {OUTPUT_DIR}")
    print(f"   Ratios         : train={TRAIN_RATIO} | val={VAL_RATIO} | test={TEST_RATIO}\n")

    # Vérification
    if not Path(DATASET_DIR).exists():
        raise FileNotFoundError(f"Dossier introuvable : {DATASET_DIR}")
    if not (Path(DATASET_DIR) / "images").exists():
        raise FileNotFoundError(f"Sous-dossier 'images/' manquant dans {DATASET_DIR}")

    # Chargement
    print("  Chargement des données...")
    samples = load_dataset(DATASET_DIR)
    print(f"  → {len(samples)} images trouvées")

    class_names = load_class_names(DATASET_DIR)
    if class_names:
        print(f"  → Classes : {list(class_names.values())}")
    else:
        print("  ⚠️  data.yaml non trouvé — les classes seront nommées class_0, class_1 ...")

    # Split
    print("\n  Calcul du split stratifié...")
    train, val, test = stratified_split(samples, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    # Copie des fichiers
    print("  Copie des fichiers...")
    copy_split(train, "train", OUTPUT_DIR)
    copy_split(val,   "val",   OUTPUT_DIR)
    copy_split(test,  "test",  OUTPUT_DIR)

    # data.yaml
    create_yaml(OUTPUT_DIR, DATASET_DIR, class_names)

    # Rapport
    print_distribution(train, val, test, class_names)

    print(f"\n✅  Split terminé ! Dossier de sortie : {OUTPUT_DIR}/")
    print("   Structure créée :")
    print("     dataset_split/")
    print("       ├── train/images/  & train/labels/")
    print("       ├── val/images/    & val/labels/")
    print("       ├── test/images/   & test/labels/")
    print("       └── data.yaml  ← utilise ce fichier pour entraîner YOLO\n")


if __name__ == "__main__":
    main()