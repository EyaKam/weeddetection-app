"""
=============================================================
  cell_grid.py — Division de l'image en cellules

  PATCH_SIZE importé depuis config.py — un seul endroit
  à modifier pour synchroniser create_test_image.py et cell_grid.py

  MODE :
    USE_PATCH_MODE = True  → patches fixes (provisoire)
    USE_PATCH_MODE = False → carré de référence manuel (production)

  UTILISATION DIRECTE :
    python scripts/cell_grid.py --image ./results/test_composite.jpg
=============================================================
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import PATCH_SIZE

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
USE_PATCH_MODE = True   # True  = patches fixes (provisoire)
                        # False = carré de référence manuel (production)
# ──────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
#  DÉTECTION AUTOMATIQUE DE LA TAILLE DES PATCHES
# ══════════════════════════════════════════════════════════════

def detect_patch_size(image):
    """
    Détecte automatiquement la taille des patches dans une image
    composite en cherchant les lignes de jointure.

    Stratégie :
      - Convertit en niveaux de gris
      - Calcule la variance par colonne et par ligne
      - Les minima de variance correspondent aux jointures entre patches
      - La distance entre jointures = taille du patch

    Si la détection échoue → retourne PATCH_SIZE depuis config.py
    """
    h, w   = image.shape[:2]
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Variance par colonne (détecte jointures verticales)
    col_var = np.var(gray, axis=0)

    # Variance par ligne (détecte jointures horizontales)
    row_var = np.var(gray, axis=1)

    def find_patch_size_from_variance(variance, total_size, min_size=64):
        """
        Cherche la période dominante dans le signal de variance
        en testant différentes tailles de patch.
        """
        best_size  = None
        best_score = float("inf")

        for size in range(min_size, total_size // 2):
            # Score = variance entre les valeurs aux multiples de size
            positions = list(range(size - 1, total_size, size))
            if len(positions) < 2:
                continue
            vals  = variance[positions]
            score = np.std(vals)
            if score < best_score:
                best_score = score
                best_size  = size

        return best_size

    try:
        # Détection sur les colonnes
        patch_w = find_patch_size_from_variance(col_var, w)
        # Détection sur les lignes
        patch_h = find_patch_size_from_variance(row_var, h)

        if patch_w and patch_h:
            detected = int(round((patch_w + patch_h) / 2))
            # Vérifie que la détection est cohérente
            if abs(patch_w - patch_h) < 50 and 64 <= detected <= 2048:
                print(f"  🔍 Taille patch détectée : {detected} px "
                      f"(col={patch_w}, row={patch_h})")
                return detected
    except Exception:
        pass

    # Fallback → valeur depuis config.py
    print(f"  ℹ️  Détection impossible → PATCH_SIZE config : {PATCH_SIZE} px")
    return PATCH_SIZE


# ══════════════════════════════════════════════════════════════
#  MODE provisoire — PATCHES FIXES
# ══════════════════════════════════════════════════════════════

def divide_into_patches(image, patch_size=None):
    """
    Divise l'image en patches carrés de taille fixe.

    Si patch_size est None :
      → tente la détection automatique depuis l'image
      → fallback sur PATCH_SIZE de config.py

    Retourne :
      cells, n_rows, n_cols, pixels_per_meter 😊 patch_size)
    """
    if patch_size is None:
        patch_size = detect_patch_size(image)

    h, w   = image.shape[:2]
    n_cols = (w + patch_size - 1) // patch_size
    n_rows = (h + patch_size - 1) // patch_size

    print(f"\n  📐 Mode provisoire — patches {patch_size}×{patch_size} px")
    print(f"     Image   : {w} × {h} px")
    print(f"     Grille  : {n_rows} lignes × {n_cols} colonnes = {n_rows*n_cols} patches")

    cells = []
    for row in range(n_rows):
        for col in range(n_cols):
            x1 = col * patch_size
            y1 = row * patch_size
            x2 = min(x1 + patch_size, w)
            y2 = min(y1 + patch_size, h)

            patch = image[y1:y2, x1:x2]

            # Padding si patch incomplet (bords)
            ph, pw = patch.shape[:2]
            if ph < patch_size or pw < patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded

            cells.append({
                "row"     : row,
                "col"     : col,
                "x1"      : x1,
                "y1"      : y1,
                "x2"      : x2,
                "y2"      : y2,
                "image"   : patch,
                "area_px" : patch_size * patch_size,
            })

    return cells, n_rows, n_cols, float(patch_size)


# ══════════════════════════════════════════════════════════════
#  MODE production — RÉFÉRENCE MANUELLE
# ══════════════════════════════════════════════════════════════

def select_reference_square(image):
    """
    Fenêtre interactive — clique sur les 4 coins du carré 1m².
    Ordre : haut-gauche → haut-droit → bas-droit → bas-gauche
    'r' = recommencer  |  'q' = quitter
    """
    print("\n  📐 Mode production — sélection carré de référence (1m²)")
    print("     1. Haut-gauche  2. Haut-droit")
    print("     3. Bas-droit    4. Bas-gauche")
    print("     'r' = recommencer  |  'q' = quitter\n")

    points      = []
    display_img = image.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal display_img
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(display_img, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(display_img, str(len(points)),
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if len(points) > 1:
                cv2.line(display_img, points[-2], points[-1], (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(display_img, points[-1], points[0], (0, 255, 0), 2)
            cv2.imshow("Selectionne le carre de reference", display_img)

    cv2.namedWindow("Selectionne le carre de reference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Selectionne le carre de reference", 1200, 800)
    cv2.setMouseCallback("Selectionne le carre de reference", mouse_callback)
    cv2.imshow("Selectionne le carre de reference", display_img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            points.clear()
            display_img = image.copy()
            cv2.imshow("Selectionne le carre de reference", display_img)
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
        elif len(points) == 4:
            cv2.waitKey(800)
            break

    cv2.destroyAllWindows()

    pts              = np.array(points, dtype=np.float32)
    widths           = [np.linalg.norm(pts[1] - pts[0]),
                        np.linalg.norm(pts[2] - pts[3])]
    heights          = [np.linalg.norm(pts[3] - pts[0]),
                        np.linalg.norm(pts[2] - pts[1])]
    pixels_per_meter = (np.mean(widths) + np.mean(heights)) / 2

    print(f"  ✅ {pixels_per_meter:.1f} px = 1 mètre")
    return points, pixels_per_meter


def divide_into_cells(image, pixels_per_meter):
    """Divise l'image selon l'échelle calibrée (Mode production)."""
    h, w      = image.shape[:2]
    cell_size = int(pixels_per_meter)
    cells     = []

    row = 0
    y   = 0
    while y < h:
        col = 0
        x   = 0
        while x < w:
            x2 = min(x + cell_size, w)
            y2 = min(y + cell_size, h)
            cells.append({
                "row"     : row,
                "col"     : col,
                "x1"      : x,
                "y1"      : y,
                "x2"      : x2,
                "y2"      : y2,
                "image"   : image[y:y2, x:x2],
                "area_px" : (x2 - x) * (y2 - y),
            })
            x   += cell_size
            col += 1
        y   += cell_size
        row += 1

    n_rows = row
    n_cols = max(c["col"] for c in cells) + 1 if cells else 0

    print(f"\n  📐 Mode production")
    print(f"     Cellule : {cell_size} × {cell_size} px")
    print(f"     Grille  : {n_rows} × {n_cols} = {len(cells)} cellules")

    return cells, n_rows, n_cols


# ══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE UNIFIÉ
# ══════════════════════════════════════════════════════════════

def run_grid(image):
    """
    Appelé par inference_density.py.
    Choisit le mode selon USE_PATCH_MODE.

    Retourne : cells, n_rows, n_cols, pixels_per_meter
    """
    if USE_PATCH_MODE:
        # patch_size=None → détection automatique depuis l'image
        return divide_into_patches(image, patch_size=None)
    else:
        _, pixels_per_meter = select_reference_square(image)
        cells, n_rows, n_cols = divide_into_cells(image, pixels_per_meter)
        return cells, n_rows, n_cols, pixels_per_meter


# ══════════════════════════════════════════════════════════════
#  LANCEMENT DIRECT — test visuel
# ══════════════════════════════════════════════════════════════

def _run_test(image_path):
    print(f"\n🌾  Test cell_grid.py")
    print(f"   Image : {image_path}")
    mode = "Patches auto" if USE_PATCH_MODE else "Référence manuelle"
    print(f"   Mode  : {mode}\n")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ❌ Image introuvable : {image_path}")
        return

    cells, n_rows, n_cols, ppm = run_grid(image)

    # Dossier de sortie
    out_dir = Path(image_path).parent.parent / "results" / "cell_grid_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde patches
    patches_dir = out_dir / "patches"
    patches_dir.mkdir(exist_ok=True)
    for cell in cells:
        name = f"patch_R{cell['row']+1:02d}_C{cell['col']+1:02d}.jpg"
        cv2.imwrite(str(patches_dir / name), cell["image"])

    # Image grille
    grid_img = image.copy()
    for cell in cells:
        cv2.rectangle(grid_img,
                      (cell["x1"], cell["y1"]),
                      (cell["x2"], cell["y2"]),
                      (0, 255, 0), 3)
        label = f"R{cell['row']+1}C{cell['col']+1}"
        cv2.putText(grid_img, label,
                    (cell["x1"] + 8, cell["y1"] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    grid_path = out_dir / "grid.jpg"
    cv2.imwrite(str(grid_path), grid_img)

    print(f"\n  ✅ Terminé !")
    print(f"     Grille   : {grid_path}")
    print(f"     Patches  : {patches_dir}  ({len(cells)} fichiers)")
    print(f"     PATCH_SIZE utilisé : {int(ppm)} px\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    _run_test(args.image)