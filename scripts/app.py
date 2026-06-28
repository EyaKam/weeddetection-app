"""
=============================================================
  app.py — Interface WeedSense
  Lance avec : streamlit run app.py
=============================================================
"""

import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import cv2
import numpy as np

# ──────────────────────────────────────────────
#  CONFIG PAGE
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="WeedDetection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --forest-dark: #1B4332;
    --forest:      #2D6A4F;
    --moss:        #52B788;
    --light:       #95D5B2;
    --pale:        #D8F3DC;
    --gray:        #6B7280;
    --border:      #C3E6CD;
  }

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
  .main { background: #F6FAF7; }
  .block-container { padding: 2rem 3rem !important; max-width: 1200px; }

  /* Header */
  .ws-header {
    background: var(--forest-dark); border-radius: 20px;
    padding: 2.5rem 3rem; margin-bottom: 2rem;
    display: flex; align-items: center; justify-content: space-between;
  }
  .ws-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem; color: white; letter-spacing: -0.03em; margin: 0;
  }
  .ws-title em { color: #95D5B2; font-style: italic; }
  .ws-sub { color: #95D5B2; font-size: 0.9rem; margin-top: 0.4rem; }
  .ws-badge {
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px; padding: 0.4rem 1rem;
    color: white; font-size: 0.78rem; font-weight: 500;
  }

  /* Upload */
  .upload-placeholder {
    border: 2px dashed var(--border); border-radius: 16px;
    padding: 2.5rem 2rem; text-align: center; background: white;
  }

  /* Stat chips */
  .stat-chip {
    border-radius: 14px; padding: 1.1rem 1rem;
    text-align: center; border: 1px solid;
  }
  .stat-chip-num {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; line-height: 1; letter-spacing: -0.04em;
  }
  .stat-chip-pct  { font-size: 0.7rem; font-weight: 500; margin-top: 0.2rem; }
  .stat-chip-label{ font-size: 0.7rem; color: var(--gray); margin-top: 0.15rem; }
  .chip-red    { background: #FEE2E2; border-color: #FCA5A5; }
  .chip-orange { background: #FFEDD5; border-color: #FDBA74; }
  .chip-yellow { background: #FEF9C3; border-color: #FDE047; }
  .chip-green  { background: #DCFCE7; border-color: var(--border); }
  .chip-red    .stat-chip-num,.chip-red    .stat-chip-pct { color: #991B1B; }
  .chip-orange .stat-chip-num,.chip-orange .stat-chip-pct { color: #9A3412; }
  .chip-yellow .stat-chip-num,.chip-yellow .stat-chip-pct { color: #854D0E; }
  .chip-green  .stat-chip-num,.chip-green  .stat-chip-pct { color: #166534; }

  /* Summary cards */
  .summary-card {
    background: var(--forest-dark); border-radius: 16px;
    padding: 1.2rem 1.6rem; margin-bottom: 0;
    display: flex; align-items: center; justify-content: space-between;
  }
  .summary-card-label { color: var(--light); font-size: 0.82rem; }
  .summary-card-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem; color: white; letter-spacing: -0.04em;
  }

  /* Species rows */
  .sp-row {
    display: flex; align-items: center; padding: 0.55rem 1rem;
    background: white; border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: 0.35rem;
    justify-content: space-between;
  }
  .sp-row-name { font-size: 0.82rem; font-weight: 600; color: var(--forest-dark); }
  .sp-row-cnt  {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem; color: var(--forest);
  }

  /* Matrix */
  .matrix-wrap { overflow-x: auto; overflow-y: auto; padding-bottom: 0.5rem; }
  .matrix-table {
    border-collapse: separate; border-spacing: 2px;
    font-family: 'DM Sans', sans-serif;
  }
  .mc {
    width: 52px; min-width: 52px; height: 52px;
    border-radius: 5px; padding: 3px 4px;
    vertical-align: top; font-size: 0.55rem;
    border: 1.5px solid rgba(0,0,0,0.08);
    transition: transform 0.15s;
  }
  .mc:hover { transform: scale(1.12); z-index: 5; position: relative; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
  .mc-red    { background: #FEE2E2; border-color: #FCA5A5; }
  .mc-orange { background: #FFEDD5; border-color: #FDBA74; }
  .mc-yellow { background: #FEF9C3; border-color: #FDE047; }
  .mc-green  { background: #F0FDF4; border-color: #86EFAC; }
  .mc-id     { font-weight: 700; font-size: 0.52rem; line-height: 1.3; }
  .mc-red    .mc-id { color: #991B1B; }
  .mc-orange .mc-id { color: #9A3412; }
  .mc-yellow .mc-id { color: #854D0E; }
  .mc-green  .mc-id { color: #166534; }
  .mc-score  { font-size: 0.52rem; color: #374151; }
  .mc-sp     { font-size: 0.48rem; color: #6B7280; line-height: 1.3; margin-top: 1px; }
  .mc-dose   { font-size: 0.5rem; font-weight: 700; margin-top: 2px; }
  .mc-red    .mc-dose { color: #991B1B; }
  .mc-orange .mc-dose { color: #9A3412; }

  /* Legend */
  .leg {
    display: flex; gap: 1rem; flex-wrap: wrap;
    margin-bottom: 0.6rem; align-items: center;
  }
  .leg-item { display: flex; align-items: center; gap: 0.3rem; }
  .leg-dot  { width: 11px; height: 11px; border-radius: 3px; }
  .leg-txt  { font-size: 0.7rem; color: var(--gray); }

  /* Divider */
  .report-divider {
    border: none; border-top: 1px solid var(--border);
    margin: 1.5rem 0;
  }
  .section-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--moss); margin-bottom: 0.7rem;
  }

  /* Buttons */
  .stButton > button {
    background: var(--forest) !important; color: white !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.65rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; width: 100% !important;
  }
  .stButton > button:hover { background: var(--forest-dark) !important; }

  .stTabs [data-baseweb="tab"] {
    background: white !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--gray) !important;
    font-weight: 500 !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--forest) !important;
    color: white !important; border-color: var(--forest) !important;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
  .stProgress > div > div > div { background: var(--moss) !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="ws-header">
  <div>
    <h1 class="ws-title">Weed<em>Detection</em></h1>
    <div class="ws-sub">Détection d'adventices · Analyse terrain · Décision pesticide</div>
  </div>
  <div class="ws-badge">🌿 Smart Agriculture</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  CHEMINS
# ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
RESULTS_DIR = PROJECT_DIR / "results" / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRIPTS_DIR))

# ──────────────────────────────────────────────
#  PARTIE HAUTE — deux colonnes
# ──────────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown("*📷 Image à analyser*")
    uploaded = st.file_uploader(
        "Upload", type=["jpg","jpeg","png"],
        label_visibility="collapsed"
    )

    if uploaded:
        img_bytes = uploaded.read()
        img_arr   = np.frombuffer(img_bytes, np.uint8)
        img_cv    = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img_rgb   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w      = img_cv.shape[:2]

        st.image(img_rgb, caption=f"{uploaded.name} · {w}×{h} px",
                 use_container_width=True)
        st.markdown("---")
        st.markdown("*⚙️ Paramètres*")
        conf = st.slider("Seuil de confiance", 0.10, 0.80, 0.30, 0.05)
        mode = st.radio("Mode division",
                        ["Patches fixes 512×512", "Référence manuelle (1m²)"])
        st.markdown("---")
        run_btn = st.button("🚀 Lancer l'analyse", type="primary")
    else:
        st.markdown("""
        <div class="upload-placeholder">
          <div style="font-size:3rem;margin-bottom:.8rem">🗺️</div>
          <div style="font-weight:600;color:#1A2E1A;margin-bottom:.3rem">Glisse ton image ici</div>
          <div style="font-size:.8rem;color:#6B7280">JPG ou PNG · Recommandé : 5472×3648</div>
        </div>""", unsafe_allow_html=True)
        run_btn = False

with col_right:
    if not uploaded:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;color:#9CA3AF">
          <div style="font-size:4rem;margin-bottom:1rem">🌾</div>
          <div style="font-size:1rem;font-weight:600;color:#374151">Upload une image pour commencer</div>
          <div style="font-size:.85rem;margin-top:.5rem">
            Le système divisera l'image en patches, détectera les espèces,
            classifiera les stades et produira une carte d'infection.
          </div>
        </div>""", unsafe_allow_html=True)

    elif run_btn:
        tmp_path = RESULTS_DIR / f"input_{datetime.now().strftime('%H%M%S')}_{uploaded.name}"
        with open(tmp_path, "wb") as f:
            f.write(img_bytes)

        prog = st.progress(0, text="Initialisation...")

        try:
            # Update patch mode
            use_patch = "True" if "Patches" in mode else "False"
            cg = SCRIPTS_DIR / "cell_grid.py"
            if cg.exists():
                txt = cg.read_text(encoding="utf-8")
                txt = txt.replace("USE_PATCH_MODE = True",  f"USE_PATCH_MODE = {use_patch}")
                txt = txt.replace("USE_PATCH_MODE = False", f"USE_PATCH_MODE = {use_patch}")
                cg.write_text(txt, encoding="utf-8")

            prog.progress(15, text="Chargement des modèles IA...")
            from cell_grid         import run_grid
            from inference_updated import load_both_models, run_inference_on_all_cells
            from features          import compute_features
            from decision          import compute_infection_score, decide_pesticide
            from visualization     import build_output_image, save_outputs
            from report            import generate_report

            model1, model2 = load_both_models()

            prog.progress(35, text="Division en cellules...")
            cells, n_rows, n_cols, ppm = run_grid(img_cv)

            prog.progress(55, text=f"Inférence sur {len(cells)} cellules...")
            all_dets = run_inference_on_all_cells(model1, model2, cells)

            prog.progress(72, text="Calcul couverture + densité...")
            cell_results = {}
            for cell in cells:
                key  = (cell["row"], cell["col"])
                dets = all_dets.get(key, [])
                cov, den = compute_features(dets, cell["area_px"])
                score    = compute_infection_score(cov, den)
                dec      = decide_pesticide(cov, den, score)
                cell_results[key] = {
                    "detections": dets, "couverture": cov,
                    "densite": den, "infection_score": score, "decision": dec,
                }

            prog.progress(88, text="Génération des images...")
            img_bboxes, img_zones, img_combined = build_output_image(
                img_cv, cells, cell_results)
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            nm    = Path(uploaded.name).stem
            paths = save_outputs(img_bboxes, img_zones, img_combined,
                                 str(RESULTS_DIR), nm, ts)

            prog.progress(96, text="Génération du rapport...")
            json_path, txt_path, rpt = generate_report(
                str(tmp_path), cells, cell_results,
                n_rows, n_cols, ppm, str(RESULTS_DIR)
            )

            prog.progress(100, text="✅ Terminé !")
            time.sleep(0.4)
            prog.empty()

            st.session_state.update({
                "paths": paths, "report": rpt, "txt_path": txt_path,
                "n_rows": n_rows, "n_cols": n_cols,
            })

        except Exception as e:
            prog.empty()
            st.error(f"❌ Erreur : {e}")
            st.exception(e)

    # Images résultat dans col_right
    if "paths" in st.session_state:
        paths = st.session_state["paths"]

        tab1, tab2, tab3 = st.tabs(["🗺️ Combiné", "🟦 Zones", "📦 Bboxes"])

        def load_img(p):
            img = cv2.imread(str(p))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

        for tab, key, label in [
            (tab1,"combined","combiné"),
            (tab2,"zones","zones"),
            (tab3,"bboxes","bboxes"),
        ]:
            with tab:
                img = load_img(paths[key])
                if img is not None:
                    st.image(img, use_container_width=True)
                    with open(paths[key], "rb") as f:
                        st.download_button(
                            f"⬇️ Télécharger {label}", f,
                            file_name=Path(paths[key]).name,
                            key=f"dl_{key}"
                        )

# ──────────────────────────────────────────────
#  PARTIE BASSE — rapport pleine largeur
# ──────────────────────────────────────────────
if "report" in st.session_state:
    rpt    = st.session_state["report"]
    n_rows = st.session_state["n_rows"]
    n_cols = st.session_state["n_cols"]
    s      = rpt["summary"]

    st.markdown("<hr class='report-divider'>", unsafe_allow_html=True)

    # ── 4 chips + 2 summary cards ──────────────
    chips_data = [
        ("chip-red",    "🔴", s["zones"].get("red",0),    s["pct_red"],    "Très infectées"),
        ("chip-orange", "🟠", s["zones"].get("orange",0), s["pct_orange"], "Infectées"),
        ("chip-yellow", "🟡", s["zones"].get("yellow",0), s["pct_yellow"], "Peu infectées"),
        ("chip-green",  "🟢", s["zones"].get("green",0),  s["pct_green"],  "Saines"),
    ]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cols6 = [c1, c2, c3, c4]
    for col, (cls, ico, val, pct, label) in zip(cols6, chips_data):
        col.markdown(f"""
        <div class="stat-chip {cls}">
          <div class="stat-chip-num">{ico} {val}</div>
          <div class="stat-chip-pct">{pct}%</div>
          <div class="stat-chip-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    c5.markdown(f"""
    <div class="summary-card">
      <div><div class="summary-card-label">Surface analysée</div>
      <div class="summary-card-val">{s['total_m2']} m²</div></div>
    </div>""", unsafe_allow_html=True)

    c6.markdown(f"""
    <div class="summary-card">
      <div><div class="summary-card-label">À pulvériser</div>
      <div class="summary-card-val">{s['nb_m2_a_pulveriser']} m²</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    pulv = s["nb_m2_a_pulveriser"]
    if pulv > 0:
        st.warning(f"⚠️  *{pulv} cellule(s)* sur {s['total_m2']} nécessitent une pulvérisation.")
    else:
        st.success("✅ Aucune zone ne nécessite de pulvérisation.")

    # ── Espèces + Matrice côte à côte ──────────
    col_sp, col_matrix = st.columns([1, 3], gap="large")

    with col_sp:
        st.markdown('<div class="section-label">Espèces détectées</div>',
                    unsafe_allow_html=True)
        if s["especes_detectees"]:
            for esp, cnt in s["especes_detectees"].items():
                st.markdown(f"""
                <div class="sp-row">
                  <div class="sp-row-name">🌿 {esp.replace("_"," ")}</div>
                  <div class="sp-row-cnt">{cnt}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Aucune adventice détectée.")

    with col_matrix:
        st.markdown('<div class="section-label">Carte terrain</div>',
                    unsafe_allow_html=True)

        # Légende
        st.markdown("""
        <div class="leg">
          <div class="leg-item"><div class="leg-dot" style="background:#FEE2E2;border:1.5px solid #FCA5A5"></div><span class="leg-txt">Très infectée</span></div>
          <div class="leg-item"><div class="leg-dot" style="background:#FFEDD5;border:1.5px solid #FDBA74"></div><span class="leg-txt">Infectée</span></div>
          <div class="leg-item"><div class="leg-dot" style="background:#FEF9C3;border:1.5px solid #FDE047"></div><span class="leg-txt">Peu infectée</span></div>
          <div class="leg-item"><div class="leg-dot" style="background:#F0FDF4;border:1.5px solid #86EFAC"></div><span class="leg-txt">Saine</span></div>
        </div>""", unsafe_allow_html=True)

        # ── Taille des cellules ─────────────────────
        CELL   = st.slider("Taille des cellules (px)", 70, 180, 90, 10,
                           key="cell_size")
        PAD    = 3
        MARGIN = 8

        # ── Génère la matrice comme image ──────────
        from PIL import Image, ImageDraw, ImageFont

        ZONE_BG = {
            "red":    (254, 226, 226),
            "orange": (255, 237, 213),
            "yellow": (254, 249, 195),
            "green":  (240, 253, 244),
        }
        ZONE_BORDER = {
            "red":    (252, 165, 165),
            "orange": (253, 186, 116),
            "yellow": (253, 224,  71),
            "green":  (134, 239, 172),
        }
        ZONE_TEXT = {
            "red":    (153,  27,  27),
            "orange": (154,  52,  18),
            "yellow": (133,  77,  14),
            "green":  ( 22, 101,  52),
        }

        img_w = MARGIN * 2 + n_cols * CELL + (n_cols - 1) * PAD
        img_h = MARGIN * 2 + n_rows * CELL + (n_rows - 1) * PAD

        canvas = Image.new("RGB", (img_w, img_h), (246, 250, 247))
        draw   = ImageDraw.Draw(canvas)

        # Tailles de police adaptées à la cellule
        fs_id    = max(8,  int(CELL * 0.13))
        fs_score = max(7,  int(CELL * 0.11))
        fs_sp    = max(6,  int(CELL * 0.10))

        try:
            fpath   = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            fpath2  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            font_id    = ImageFont.truetype(fpath,  fs_id)
            font_score = ImageFont.truetype(fpath2, fs_score)
            font_sp    = ImageFont.truetype(fpath2, fs_sp)
        except Exception:
            font_id = font_score = font_sp = ImageFont.load_default()

        cell_map = {c["id"]: c for c in rpt["cells"]}

        for r in range(n_rows):
            for c in range(n_cols):
                cell_id = f"R{r+1}C{c+1}"
                cell    = cell_map.get(cell_id)
                zone    = cell.get("zone", "green") if cell else "green"

                x1 = MARGIN + c * (CELL + PAD)
                y1 = MARGIN + r * (CELL + PAD)
                x2 = x1 + CELL
                y2 = y1 + CELL

                bg  = ZONE_BG.get(zone,    (240, 253, 244))
                brd = ZONE_BORDER.get(zone, (134, 239, 172))
                tc  = ZONE_TEXT.get(zone,   ( 22, 101,  52))

                draw.rectangle([x1, y1, x2, y2], fill=bg, outline=brd, width=2)

                if cell:
                    score = cell.get("infection_score", 0.0)
                    couv  = cell.get("couverture", {})
                    dec   = cell.get("decision", {})
                    weeds = {k: v for k, v in couv.items() if k != "ble_dur"}

                    ty = y1 + 4
                    # ID
                    draw.text((x1+5, ty), cell_id, font=font_id, fill=tc)
                    ty += fs_id + 3
                    # Score
                    draw.text((x1+5, ty), f"Score: {score:.2f}", font=font_score, fill=(80, 80, 80))
                    ty += fs_score + 3
                    # Espèces — sans emoji ni symbole spécial
                    for sp, cnt in list(weeds.items())[:2]:
                        short = sp.replace("_", " ").split(" ")[0][:9]
                        draw.text((x1+5, ty), f"{short} x{cnt}", font=font_sp, fill=(100, 100, 100))
                        ty += fs_sp + 3
                    # Dose — texte simple, pas d'emoji
                    dose = dec.get("herbicide_dose")
                    if dose:
                        dose_clean = dose.replace("L/ha", " L/ha")
                        draw.text((x1+5, y2 - fs_sp - 6),
                                  f"Dose: {dose_clean}",
                                  font=font_sp, fill=tc)

        matrix_img = np.array(canvas)
        st.image(matrix_img, use_container_width=True)

    # ── Téléchargement ──────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    txt_path = st.session_state["txt_path"]
    if Path(txt_path).exists():
        with open(txt_path, "rb") as f:
            st.download_button(
                "⬇️ Télécharger le rapport complet (TXT)",
                f, file_name=Path(txt_path).name,
                use_container_width=True
            )