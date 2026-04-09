# Deployment Checklist

## File placement

```
resume_screening_system/          ← backend root (all .py files)
├── requirements.txt              ← this file
├── Procfile                      ← this file
├── runtime.txt                   ← this file
├── render.yaml                   ← this file
├── .env.example                  ← this file
├── .gitignore                    ← this file (repo root)
└── resume_ui/
    └── frontend/
        ├── vercel.json           ← this file
        └── src/App.jsx
```

---

## Backend → Render

1. Push repo to GitHub.
2. New Web Service → connect repo → set **Root Directory** to `.` (where `app.py` lives).
3. Render auto-detects `requirements.txt` and `Procfile`.
4. Add environment variable:
   - `CORS_ORIGINS` = `https://your-app.vercel.app` (fill in after Vercel deploy)
5. Deploy. **First boot is slow** (~3–5 min) — sentence-transformers downloads the
   default model (~420 MB) into the ephemeral cache on first request.

> ⚠️ Free tier sleeps after 15 min of inactivity. First request after sleep re-loads
> the model from disk cache (~30 s). Upgrade to Starter plan + add a persistent disk
> (uncomment `disk:` block in `render.yaml`) to keep models warm across deploys.

---

## Frontend → Vercel

1. Import repo in Vercel → set **Root Directory** to `resume_ui/frontend`.
2. Framework preset: **Vite** (auto-detected).
3. Add environment variable:
   - `VITE_API_URL` = `https://your-backend.onrender.com`
4. **Update `App.jsx`** — replace the hardcoded `BASE` constant:

```js
// Before
const BASE = "http://localhost:5001";

// After
const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:5001";
```

5. Deploy.

---

## Local dev

```bash
# Backend
cd resume_screening_system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py          # http://localhost:5001

# Frontend
cd resume_ui/frontend
npm install
npm run dev            # http://localhost:5173
```
