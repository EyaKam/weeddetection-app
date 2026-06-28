"""
=============================================================
  create_test_image.py — Assemblage avec tri par couleur

  Résultat :
    Image beaucoup plus homogène visuellement
=============================================================
"""

import random
import cv2
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
IMAGES_DIR   = "./data/notaugdata/images"
OUTPUT_PATH  = "./results/test_composite2.jpg"

TARGET_W     = 5472
TARGET_H     = 3648

PATCH_SIZE   = 512
RANDOM_SEED  = 42
# ──────────────────────────────────────────────


def compute_dominant_color(img):
    """Retourne une couleur dominante simple (HSV)"""
    small = cv2.resize(img, (50, 50))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # Moyenne HSV
    mean = hsv.mean(axis=(0, 1))
    return mean  # (H, S, V)


def main():
    random.seed(RANDOM_SEED)

    images_dir = Path(IMAGES_DIR)
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_images = list(images_dir.glob("*")) 

    if not all_images:
        print(f"❌ Aucune image trouvée dans {IMAGES_DIR}")
        return

    print(f"\n🖼️  Création image composite (tri couleur)")
    print(f"   Images : {len(all_images)}")

    # ─────────────────────────────
    # 1. Charger + calcul couleur
    # ─────────────────────────────
    image_data = []
    print("  Analyse des couleurs...")

    for path in all_images:
        img = cv2.imread(str(path))
        if img is None:
            continue

        color = compute_dominant_color(img)
        image_data.append((path, color))

    if not image_data:
        print("❌ Impossible de charger les images")
        return

    # ─────────────────────────────
    # 2. Trier par couleur
    # ─────────────────────────────
    # Tri par Hue puis luminosité → transitions naturelles
    image_data.sort(key=lambda x: (x[1][0], x[1][2]))

    sorted_paths = [x[0] for x in image_data]

    # ─────────────────────────────
    # 3. Grille
    # ─────────────────────────────
    n_cols = (TARGET_W + PATCH_SIZE - 1) // PATCH_SIZE
    n_rows = (TARGET_H + PATCH_SIZE - 1) // PATCH_SIZE
    n_needed = n_rows * n_cols

    print(f"   Grille : {n_rows} × {n_cols}")

    # Répéter si pas assez d’images
    selected = []
    while len(selected) < n_needed:
        selected.extend(sorted_paths)

    selected = selected[:n_needed]

    # 👉 IMPORTANT : organisation en "gradient"
    # On remplit ligne par ligne avec alternance
    ordered = []
    for r in range(n_rows):
        row_imgs = selected[r*n_cols:(r+1)*n_cols]

        # zig-zag pour éviter coupures verticales
        if r % 2 == 1:
            row_imgs = row_imgs[::-1]

        ordered.extend(row_imgs)

    # ─────────────────────────────
    # 4. Création canvas
    # ─────────────────────────────
    canvas = np.ones((n_rows * PATCH_SIZE, n_cols * PATCH_SIZE, 3),
                     dtype=np.uint8) * 255

    print("  Assemblage...")

    loaded = 0
    failed = 0

    for idx, img_path in enumerate(ordered):
        row = idx // n_cols
        col = idx % n_cols

        img = cv2.imread(str(img_path))

        if img is None:
            failed += 1
            patch = np.full((PATCH_SIZE, PATCH_SIZE, 3), 80, dtype=np.uint8)
        else:
            patch = cv2.resize(img, (PATCH_SIZE+2, PATCH_SIZE+2))
            loaded += 1

        y1 = row * PATCH_SIZE
        x1 = col * PATCH_SIZE
        canvas[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] = patch[:PATCH_SIZE, :PATCH_SIZE]

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1:>4}/{n_needed}]...", end="\r")

    # Crop final
    canvas = canvas[:TARGET_H, :TARGET_W]

    # Sauvegarde
    cv2.imwrite(str(output_path), canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\n\n✅ Image créée (beaucoup plus homogène)")
    print(f"   Fichier : {output_path}")
    print(f"   Patches : {loaded} / {failed}")

    print("\n👉 Test maintenant ton cell_grid.py")


if __name__ == "__main__":
    main()