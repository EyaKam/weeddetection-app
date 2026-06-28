"""
=============================================================
  Fusion des classes YOLO — Sans Roboflow Pro
  Projet : Détection blé / adventices
=============================================================
  UTILISATION :
    1. Exporte ton dataset depuis Roboflow (format YOLOv8)
    2. Place l'export dans data/raw/
    3. Lance : python scripts/remap_classes.py
=============================================================
"""

import os
import shutil
import yaml
from pathlib import Path

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
INPUT_DIR  = "./data/raw"       # export Roboflow original (36 classes)
OUTPUT_DIR = "./data/remapped"  # dataset fusionné (~8 classes)

# ──────────────────────────────────────────────
#  TABLE DE FUSION
#  "ancien nom dans Roboflow" : "nouveau nom"
#  ← Vérifie que les noms correspondent EXACTEMENT
#    à ce qui est dans ton data.yaml
# ──────────────────────────────────────────────
CLASS_MAPPING = {
    # Blé dur
    "ble dur s1"        : "ble_dur",
    "ble dur s2"        : "ble_dur",
    "ble dur s3"        : "ble_dur",

    # Chrysanthème
    "chrysantheme s1"   : "chrysantheme",
    "chrysantheme s2"   : "chrysantheme",
    "chrysantheme s 3"  : "chrysantheme",
    "chrysantheme s 4"  : "chrysantheme",

    # Oxalis
    "oxalis s1"         : "oxalis",
    "oxalis s2"         : "oxalis",
    "oxalis s3"         : "oxalis",
    "oxalis s4"         : "oxalis",

    # Laitue sauvage
    "laitue sauvage s1" : "laitue_sauvage",
    "laitue sauvage s2" : "laitue_sauvage",
    "laitue sauvage s3" : "laitue_sauvage",
    "laitue sauvage s4" : "laitue_sauvage",

    # Chardon marie + chardon (même plante)
    "chardon marie s1"  : "chardon_marie",
    "chardon marie s2"  : "chardon_marie",
    "chardon marie s3"  : "chardon_marie",
    "chardon marie s4"  : "chardon_marie",
    
    # Ray gras
    "ray gras s1"       : "ray_gras",
    "ray gras s2"       : "ray_gras",
    "ray grass s3"      : "ray_gras",
    "ray grass s4"      : "ray_gras",

    # Bette sauvage
    "bette sauvage s1"  : "bette_sauvage",
    "bette sauvage s2"  : "bette_sauvage",

    # Oseille crépue
    "oseille crepue s1" : "oseille_crepue",
    "oseille crepue s2" : "oseille_crepue",

    # Pissenlit
    "pisenlit s1"       : "pisenlit",
    "pisenlit s2"       : "pisenlit",
    "pisenlit s3"       : "pisenlit",

    # Carotte sauvage
    "carotte sauvage s1": "carotte_sauvage",
    "carotte sauvage s2": "carotte_sauvage",
    "carotte sauvage s3": "carotte_sauvage",

    # ── Supprimées (trop peu d'images) ──
    "bacopa s2"         : None,
    "bacopa s3"         : None,
    "chelidoine s1"     : None,
    "coriandre s1"      : None,
    "courge 2"          : None,
    "faune sauvage s1"  : None,
    "faune sauvage s2"  : None,
    "faune sauvage s3"  : None,
    "lamier s3"         : None,
    "navet potager s1"  : None,
    "navet potager s2"  : None,
    "persil s2"         : None,
    "sedum S2"          : None,
}
# ──────────────────────────────────────────────


def load_class_names(dataset_dir):
    """Lit les noms de classes depuis data.yaml."""
    yaml_path = Path(dataset_dir) / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml introuvable dans {dataset_dir}")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return names


def build_new_classes(old_names, mapping):
    """
    Construit la liste des nouvelles classes
    et le mapping old_id → new_id.
    """
    new_classes_ordered = []
    seen = {}

    for old_id, old_name in sorted(old_names.items()):
        new_name = mapping.get(old_name)
        if new_name is None:
            continue  # classe supprimée
        if new_name not in seen:
            seen[new_name] = len(new_classes_ordered)
            new_classes_ordered.append(new_name)

    # old_id → new_id
    id_map = {}
    for old_id, old_name in old_names.items():
        new_name = mapping.get(old_name)
        if new_name is not None and new_name in seen:
            id_map[old_id] = seen[new_name]
        # else: None → classe supprimée, pas dans id_map

    return new_classes_ordered, id_map


def remap_label_file(src_path, dst_path, id_map):
    """
    Relit un fichier .txt YOLO et remplace les class_id.
    Ignore les annotations dont la classe est supprimée.
    """
    lines_out = []
    with open(src_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_id = int(parts[0])
            if old_id in id_map:
                parts[0] = str(id_map[old_id])
                lines_out.append(" ".join(parts))

    if lines_out:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w") as f:
            f.write("\n".join(lines_out) + "\n")
        return True
    return False  # toutes les annotations supprimées → image ignorée


def process_flat(input_dir, output_dir, id_map):
    """Traite un dataset plat images/ + labels"""
    src_img = Path(input_dir)  / "images"
    src_lbl = Path(input_dir)  / "labels"
    dst_img = Path(output_dir) / "images"
    dst_lbl = Path(output_dir) / "labels"

    if not src_img.exists():
        raise FileNotFoundError(f"Dossier images/ introuvable dans {input_dir}")

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    images = list(src_img.glob("*.jpg")) + list(src_img.glob("*.png"))
    kept = skipped = 0

    for img_path in images:
        lbl_path = src_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        dst_lbl_path = dst_lbl / lbl_path.name
        has_annotations = remap_label_file(lbl_path, dst_lbl_path, id_map)
        if has_annotations:
            shutil.copy2(img_path, dst_img / img_path.name)
            kept += 1
        else:
            skipped += 1

    return kept, skipped

def create_yaml(output_dir, new_classes):
    """Génère le data.yaml pour le dataset fusionné."""
    yaml_content = {
        "path"  : str(Path(output_dir).resolve()),
        "train" : "train/images",
        "val"   : "val/images",
        "test"  : "test/images",
        "nc"    : len(new_classes),
        "names" : new_classes,
    }
    out_path = Path(output_dir) / "data.yaml"
    with open(out_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    return out_path


def main():
    print("\n🌿  Fusion des classes — Weed Detection")
    print(f"   Source  : {INPUT_DIR}")
    print(f"   Sortie  : {OUTPUT_DIR}\n")

    # Chargement des classes originales
    old_names = load_class_names(INPUT_DIR)
    print(f"  Classes originales ({len(old_names)}) :")
    for old_id, old_name in sorted(old_names.items()):
        mapped = CLASS_MAPPING.get(old_name)
        if old_name not in CLASS_MAPPING:
            status = "❓ NON MAPPÉE"
        elif mapped is None:
            status = "✗  supprimée"
        else:
            status = "→ " + mapped
        print(f"    [{old_id:2}] {old_name:<25} {status}")

    # Vérification classes non mappées
    unmapped = [n for n in old_names.values() if n not in CLASS_MAPPING]
    if unmapped:
        print(f"\n  ⚠️  Classes non mappées :")
        for n in unmapped:
            print(f'       "{n}" : "nouvelle_classe",')
        print()

    # Construction du mapping
    new_classes, id_map = build_new_classes(old_names, CLASS_MAPPING)
    print(f"\n  Nouvelles classes ({len(new_classes)}) : {new_classes}")

    # Traitement des splits
    # Traitement
    print("\n  Traitement des fichiers...")
    kept, skipped = process_flat(INPUT_DIR, OUTPUT_DIR, id_map)
    print(f"  Images conservées : {kept}")
    print(f"  Images ignorées   : {skipped}")

    # data.yaml
    yaml_path = create_yaml(OUTPUT_DIR, new_classes)
    print(f"\n  data.yaml créé → {yaml_path}")

    print(f"""
  ═══════════════════════════════════════
    RÉSUMÉ
  ═══════════════════════════════════════
    Classes originales  : {len(old_names)}
    Nouvelles classes   : {len(new_classes)}
    Images conservées   : {kept}
    Images supprimées   : {skipped}
    Dossier de sortie   : {OUTPUT_DIR}/
  ═══════════════════════════════════════

  ✅ Terminé !
     1. Lance : python main.py --phase 2
     2. Puis upload data/remapped/ sur Kaggle
""")


if __name__ == "__main__":
    main()