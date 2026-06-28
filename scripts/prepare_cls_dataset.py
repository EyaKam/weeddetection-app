"""
=============================================================
  prepare_cls_dataset.py — Préparation dataset classification
  Modèle 2 : Classification des stades

  Dataset source : data/raw/ (46 classes, images 512×512)
  Dataset sortie : data/dataset_cls/ (33 classes avec stades)
  Structure      : train/val/test par dossier classe

  UTILISATION :
    python scripts/prepare_cls_dataset.py
=============================================================
"""

import random
import shutil
import yaml
from collections import defaultdict
from pathlib import Path

import cv2

# ──────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────
RAW_DIR     = "./data/raw"         # export Roboflow original (46 classes)
OUTPUT_DIR  = "./data/dataset_cls" # dataset de classification

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10
RANDOM_SEED = 42

CROP_SIZE   = 224    # adapté aux images 512×512
BBOX_MARGIN = 0.10   # 10% de marge autour de la bbox

# ──────────────────────────────────────────────────────────────
#  CLASSES À IGNORER (pas de stade réel ou trop peu d'images)
# ──────────────────────────────────────────────────────────────
IGNORED_CLASSES = {
    "bacopa s2",
    "bacopa s3",
    "chelidoine s1",
    "coriandre s1",
    "courge 2",
    "faune sauvage s1",
    "faune sauvage s2",
    "faune sauvage s3",
    "lamier s3",
    "navet potager s1",
    "navet potager s2",
    "persil s2",
    "sedum S2",
}
# ──────────────────────────────────────────────────────────────
#  FUSION DES CLASSES RARES
# ──────────────────────────────────────────────────────────────
CLASS_MERGE = {
    "ray_gras_s4": "ray_gras_s3",
    "carotte_sauvage_s3": "carotte_sauvage_s2",
}

# ──────────────────────────────────────────────────────────────
#  NORMALISATION DES NOMS
#  Gère les cas spéciaux comme "chrysantheme s 3" (espace avant chiffre)
# ──────────────────────────────────────────────────────────────
def normalize_class_name(name):
    """
    Convertit le nom Roboflow → nom de dossier propre.
    Exemples :
      "chrysantheme s 3"  → "chrysantheme_s3"
      "laitue sauvage s1" → "laitue_sauvage_s1"
      "ray grass s3"      → "ray_gras_s3"   (correction orthographe)
    """
    n = name.strip().lower()

    # Correction : "ray grass" → "ray_gras" (cohérence avec Modèle 1)
    n = n.replace("ray grass", "ray_gras")
    n = n.replace("ray gras",  "ray_gras")

    # Supprime l'espace entre "s" et le chiffre : "s 3" → "s3"
    import re
    n = re.sub(r's\s+(\d)', r's\1', n)

    # Remplace les espaces restants par des underscores
    n = n.replace(" ", "_")

    return n


# ──────────────────────────────────────────────────────────────
#  CHARGEMENT DES CLASSES
# ──────────────────────────────────────────────────────────────
def load_class_names(raw_dir):
    yaml_path = Path(raw_dir) / "data.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    return {i: n for i, n in enumerate(names)} if isinstance(names, list) else names


# ──────────────────────────────────────────────────────────────
#  CROP D'UNE BBOX
# ──────────────────────────────────────────────────────────────
def crop_bbox(image, x_center, y_center, w, h, margin=BBOX_MARGIN):
    """
    Convertit les coordonnées YOLO normalisées en pixels
    et découpe la région avec une marge autour.

    Format YOLO : x_center, y_center, width, height ∈ [0, 1]

    Retourne le crop redimensionné à CROP_SIZE × CROP_SIZE,
    ou None si le crop est invalide.
    """
    img_h, img_w = image.shape[:2]

    # Conversion normalisé → pixels
    cx = int(x_center * img_w)
    cy = int(y_center * img_h)
    bw = int(w * img_w)
    bh = int(h * img_h)

    # Marge autour de la bbox
    mx = int(bw * margin)
    my = int(bh * margin)

    # Coordonnées finales avec clamp aux bords de l'image
    x1 = max(0, cx - bw // 2 - mx)
    y1 = max(0, cy - bh // 2 - my)
    x2 = min(img_w, cx + bw // 2 + mx)
    y2 = min(img_h, cy + bh // 2 + my)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                      interpolation=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────────
#  EXTRACTION DE TOUS LES CROPS
# ──────────────────────────────────────────────────────────────
def extract_all_crops(raw_dir, class_names):
    """
    Parcourt toutes les images + labels du dataset brut.
    Pour chaque annotation valide, extrait le crop de la bbox.

    Retourne :
      crops_by_class : {class_name_normalisé: [(crop_img, source_stem)]}
    """
    images_dir = Path(raw_dir) / "images"
    labels_dir = Path(raw_dir) / "labels"

    image_files = (list(images_dir.glob("*.jpg")) +
                   list(images_dir.glob("*.png")))

    print(f"  {len(image_files)} images à traiter...")

    crops_by_class = defaultdict(list)
    n_extracted = 0
    n_ignored   = 0
    n_invalid   = 0

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue

            class_id   = int(parts[0])
            class_name = class_names.get(class_id, f"class_{class_id}")

            # Ignore les classes sans stade réel
            if class_name in IGNORED_CLASSES:
                n_ignored += 1
                continue

            x_center, y_center, w, h = map(float, parts[1:5])

            crop = crop_bbox(image, x_center, y_center, w, h)
            if crop is None:
                n_invalid += 1
                continue

            # Normalisation du nom
            norm_name = normalize_class_name(class_name)

            # Fusion des classes rares
            if norm_name in CLASS_MERGE:
                norm_name = CLASS_MERGE[norm_name]

            # Sauvegarde du crop
            crops_by_class[norm_name].append((crop, img_path.stem))

            n_extracted += 1

    print(f"  ✅ {n_extracted} crops extraits")
    print(f"     {n_ignored} ignorés (classes sans stade)")
    print(f"     {n_invalid} invalides (bbox trop petite)")
    return crops_by_class


# ──────────────────────────────────────────────────────────────
#  SPLIT STRATIFIÉ + SAUVEGARDE
# ──────────────────────────────────────────────────────────────
def split_and_save(crops_by_class, output_dir):
    """
    Split stratifié train/val/test et sauvegarde les crops
    dans la structure attendue par YOLOv8-cls.
    """
    output_path = Path(output_dir)
    stats       = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for class_name, crops in sorted(crops_by_class.items()):
        random.shuffle(crops)
        n = len(crops)

        if n < 3:
            print(f"  ⚠️  {class_name} : {n} crop(s) seulement → ignorée")
            continue

        n_train = max(1, round(n * TRAIN_RATIO))
        n_val   = max(1, round(n * VAL_RATIO))
        n_test  = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val  = n - n_train

        splits = {
            "train": crops[:n_train],
            "val"  : crops[n_train: n_train + n_val],
            "test" : crops[n_train + n_val:],
        }

        for split_name, split_crops in splits.items():
            split_dir = output_path / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for i, (crop_img, source) in enumerate(split_crops):
                out_path = split_dir / f"{source}_{i:04d}.jpg"
                cv2.imwrite(str(out_path), crop_img)
                stats[class_name][split_name] += 1

    return stats


# ──────────────────────────────────────────────────────────────
#  AFFICHAGE DU RÉSUMÉ
# ──────────────────────────────────────────────────────────────
def print_summary(stats, output_dir):
    print("\n" + "=" * 68)
    print("  DATASET CLASSIFICATION — DISTRIBUTION")
    print("=" * 68)
    print(f"  {'Classe':<32} {'TRAIN':>7} {'VAL':>6} {'TEST':>6} {'TOTAL':>7}")
    print("-" * 68)

    total_train = total_val = total_test = 0

    for class_name, counts in sorted(stats.items()):
        t  = counts["train"]
        v  = counts["val"]
        te = counts["test"]
        tt = t + v + te
        total_train += t
        total_val   += v
        total_test  += te

        # Avertissement si classe sous-représentée
        warn = " ⚠️ " if tt < 20 else ""
        print(f"  {class_name:<32} {t:>7} {v:>6} {te:>6} {tt:>7}{warn}")

    print("-" * 68)
    grand_total = total_train + total_val + total_test
    print(f"  {'TOTAL':<32} {total_train:>7} {total_val:>6} {total_test:>6} {grand_total:>7}")
    print("=" * 68)
    print(f"\n  Classes valides : {len(stats)}/33")
    print(f"  Crops totaux    : {grand_total}")
    print(f"  Taille des crops: {CROP_SIZE}×{CROP_SIZE} px")
    print(f"  Dossier sortie  : {output_dir}/")
    print(f"""
  Prochaines étapes :
    1. Vérifie les classes avec ⚠️  (moins de 20 crops)
    2. Zippe le dossier :
       Compress-Archive -Path data/dataset_cls -DestinationPath dataset_cls.zip
    3. Upload sur Kaggle comme Dataset "weed-cls"
    4. Lance le notebook kaggle_model2_stages.ipynb
""")


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print("\n🌿  Préparation dataset classification — Stades")
    print(f"   Source  : {RAW_DIR}  (images 512×512)")
    print(f"   Sortie  : {OUTPUT_DIR}")
    print(f"   Crop    : {CROP_SIZE}×{CROP_SIZE} px")
    print(f"   Classes : 33 avec stades / 13 ignorées\n")

    random.seed(RANDOM_SEED)

    # Vérifications
    if not (Path(RAW_DIR) / "images").exists():
        print(f"  ❌ Dossier images/ introuvable dans {RAW_DIR}")
        return
    if not (Path(RAW_DIR) / "data.yaml").exists():
        print(f"  ❌ data.yaml introuvable dans {RAW_DIR}")
        return

    # Nettoyage
    if Path(OUTPUT_DIR).exists():
        print(f"  🗑️  Suppression de l'ancien dossier : {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Chargement classes
    class_names   = load_class_names(RAW_DIR)
    valid_classes = {k: v for k, v in class_names.items()
                     if v not in IGNORED_CLASSES}
    print(f"  Classes totales : {len(class_names)}")
    print(f"  Classes valides : {len(valid_classes)}")
    print(f"  Classes ignorées: {len(IGNORED_CLASSES)}\n")

    # Extraction des crops
    print("  Extraction des crops...")
    crops_by_class = extract_all_crops(RAW_DIR, class_names)

    # Split + sauvegarde
    print("\n  Split et sauvegarde...")
    stats = split_and_save(crops_by_class, OUTPUT_DIR)

    # Résumé
    print_summary(stats, OUTPUT_DIR)


if __name__ == "__main__":
    main()