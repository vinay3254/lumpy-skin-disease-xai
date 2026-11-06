# LSD Web Frontend — Design

Local-only web app for the viva demo. Backend reuses existing `src/` code.

## Backend: `api.py` (FastAPI)

- Loads runs lazily via `src.evaluate.load_run(run)`; caches models in a dict.
- Default runs: binary = best `*_c2*` hybrid by test f1 in `metrics.json`; 3-class = `hybrid_attention_c3`.
- CORS allow `http://localhost:5173`.
- Endpoints:
  - `GET /api/models` → `[{run, classes, mode, fusion, metrics}]` for every run dir with `best.pt` + `metrics.json`; plus `{"defaults": {"2": run, "3": run}}` — response shape `{"runs": [...], "defaults": {...}}`.
  - `POST /api/predict` multipart `file`, form `run` (optional; default binary) → `{run, class_names, probs: {name: p}, pred, confidence, gradcam: "data:image/png;base64,...", lime: "data:..."}`. LIME num_samples 300.
  - `POST /api/shap` multipart `file`, form `run` → `{shap: "data:..."}` (background: 20 train images, cached per run).
  - `GET /api/results` → `{ablation: [rows from outputs/ablation.csv], figures: {run: {confusion, roc, curves} as /static URLs}}`.
  - `GET /api/samples` → list of `/static/samples/<file>` URLs. On startup, if `outputs/samples/` is empty, copy 3 test images per class from `data/splits_c3.csv` there.
  - Static mount `/static` → `outputs/`.
- Errors: non-image upload → 400 `{"detail": "Invalid image"}`; unknown run → 404.
- GPU access guarded by a `threading.Lock`.

## Frontend: `frontend/` (React + Vite + TypeScript + Tailwind v4)

- API base from `import.meta.env.VITE_API_URL` default `http://localhost:8000`.
- Layout: top nav with project title "LSD-XAI: Hybrid CNN–Transformer for Lumpy Skin Disease"; tabs Diagnose / Results / About. Clean, light, green-accent medical/agri look; responsive.
- Diagnose: drag-and-drop + file picker; sample image strip (click to use); model toggle Binary / 3-class; Analyze button with loading state; result card (pred label badge colored: healthy green, lsd red, other amber; confidence %; per-class bars); XAI grid with original, Grad-CAM, LIME; "Explain deeper (SHAP)" button loads SHAP panel; error toast.
- Results: ablation table (run, params M, accuracy, precision, recall, f1, roc_auc; best row highlighted); recharts bar chart of accuracy & f1 per run; per-run figure gallery (select run → confusion matrix, ROC, curves).
- About: 6-step pipeline cards (Data Acquisition → Preprocessing → Dataset Preparation → Hybrid CNN–Transformer → Classification → XAI → Evaluation); dataset table (warcoder Lumpy Skin Images: 700 healthy / 324 LSD; Cattle Diseases: 1291 healthy / 1207 LSD / 746 foot-and-mouth); architecture text (EfficientNet-B0 + DeiT-Tiny, concat / cross-attention fusion); note on pHash dedupe preventing train/test leakage; disclaimer "Research prototype, not a veterinary diagnostic tool".
- Code split: `src/api.ts` (typed client), `src/types.ts`, `src/pages/{Diagnose,Results,About}.tsx`, `src/components/*`.

## Run

`start.sh`: starts `uvicorn api:app --port 8000` and `npm --prefix frontend run dev`.

## Tests

- `tests/test_api.py` with FastAPI TestClient: `/api/models` 200; `/api/predict` on a generated image returns probs summing ≈1 and gradcam data URL; non-image → 400. Skip if no trained run exists.
- `npm run build` passes (tsc + vite).
