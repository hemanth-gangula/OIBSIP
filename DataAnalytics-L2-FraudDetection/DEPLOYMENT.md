# Deployment Guide

**Credit Card Fraud Detection — Oasis Infobyte L2 Task 3**

---

## Architecture Overview

This project has two independently deployed components:

```
┌──────────────────────────────────────────────────────────────┐
│                        USER BROWSER                          │
└────────────────────────────┬─────────────────────────────────┘
                             │ HTTPS
┌────────────────────────────▼─────────────────────────────────┐
│              VERCEL  (Frontend)                              │
│  Next.js 14 · Static pages · Tailwind · Recharts            │
│  https://your-project.vercel.app                            │
└────────────────────────────┬─────────────────────────────────┘
                             │ REST API  (NEXT_PUBLIC_API_URL)
┌────────────────────────────▼─────────────────────────────────┐
│         RENDER / RAILWAY  (Python Backend)                   │
│  FastAPI · uvicorn · scikit-learn · joblib models           │
│  https://your-backend.onrender.com                          │
└──────────────────────────────────────────────────────────────┘
```

> **Why two separate services?**
> Vercel's serverless functions have a hard 50 MB uncompressed size limit per function.
> A scikit-learn Random Forest with 100 trees serialised via joblib is typically 40–80 MB
> *before* adding numpy, pandas, and scipy. This combination **cannot run on Vercel**.
> The Python ML backend must be hosted on a service that supports persistent Python
> runtimes — Render, Railway, Hugging Face Spaces, or a VPS are all correct choices.

---

## Part 1 — Local Development

### Prerequisites

| Tool | Minimum Version | Check |
|------|----------------|-------|
| Python | 3.10 | `python --version` |
| pip | 23+ | `pip --version` |
| Node.js | 18 | `node --version` |
| npm | 9+ | `npm --version` |
| Git | any | `git --version` |

---

### 1.1 Clone the Repository

```bash
git clone https://github.com/<your-username>/OIBSIP.git
cd OIBSIP/DataAnalytics-L2-FraudDetection
```

---

### 1.2 Download the Dataset

The dataset is excluded from git (144 MB). Download it from Kaggle:

1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Sign in (free account) and click **Download**
3. Extract the zip — you will get `creditcard.csv`
4. Place it at:

```
data/creditcard.csv
```

Verify:
```bash
# Windows PowerShell
(Get-Item "data\creditcard.csv").Length / 1MB   # should show ~143.84

# macOS / Linux
du -sh data/creditcard.csv   # should show ~144M
```

---

### 1.3 Install Python Dependencies

```bash
pip install -r requirements.txt
```

Installs: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn,
joblib, fastapi, uvicorn, pydantic, jupyter, httpx.

---

### 1.4 Run the ML Training Pipeline

```bash
# Windows
run_training.bat

# macOS / Linux
python run_training.py
```

Expected output (abridged):
```
====== Training Logistic Regression ======
  Training time: ~12s
  Precision: 0.0557  Recall: 0.9184  F1: 0.1050  AUC: 0.9706

====== Training Random Forest ======
  Training time: ~264s
  Precision: 0.8247  Recall: 0.8163  F1: 0.8205  AUC: 0.9683

====== Training Complete ======
```

Artifacts created in `models/`:
```
models/
├── logistic_regression.pkl
├── random_forest.pkl
├── scaler.pkl
├── feature_columns.json
├── dataset_stats.json
├── model_metrics.json
├── lr_coefficients.json
├── rf_feature_importance.json
└── sample_transactions.json
```

---

### 1.5 Generate Evaluation Charts

```bash
# Windows
run_evaluation.bat

# macOS / Linux
python run_evaluation.py
```

Generates 13 real charts in `models/plots/` and copies them to `screenshots/`.

---

### 1.6 Start the FastAPI Backend

```bash
cd app/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Verify it is running:
```bash
curl http://localhost:8000/
# {"service":"Fraud Detection API","status":"online","version":"1.0.0",...}

curl http://localhost:8000/api/stats
# {"status":"ok","data":{"total_transactions":284807,"fraud_count":492,...}}
```

Interactive API docs: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc

---

### 1.7 Start the Next.js Frontend

```bash
cd app/frontend

# Copy the example env file
cp .env.local.example .env.local
# Default value: NEXT_PUBLIC_API_URL=http://localhost:8000

npm install
npm run dev
```

Open http://localhost:3000

Pages:
- `/`            — Dashboard with dataset stats and model overview
- `/eda`         — Exploratory data analysis and class imbalance
- `/performance` — Model metrics, confusion matrices, ROC curves
- `/predict`     — Live transaction fraud prediction

---

### 1.8 Open the Jupyter Notebook (optional)

```bash
jupyter notebook notebooks/fraud_detection_analysis.ipynb
```

Run all cells top to bottom. The notebook uses paths relative to the
`notebooks/` directory and loads models from `../models/`.

---

### 1.9 Run Backend Tests

```bash
# From project root
python test_backend.py
```

Expected: `RESULTS: 46 passed, 0 failed`

---

## Part 2 — Vercel Deployment (Frontend)

Vercel hosts **only the Next.js frontend**. It does not run the Python backend.

### Step-by-step

**1. Push to GitHub**

```bash
cd OIBSIP/DataAnalytics-L2-FraudDetection
git init
git add .
git commit -m "feat: complete fraud detection project"
git remote add origin https://github.com/<your-username>/OIBSIP.git
git push -u origin main
```

**2. Import project in Vercel**

- Go to https://vercel.com/new
- Click **Import Git Repository** → select your `OIBSIP` repo
- Set **Root Directory** to `DataAnalytics-L2-FraudDetection/app/frontend`
- Framework preset: **Next.js** (auto-detected)

**3. Set the environment variable**

In Vercel project settings → **Environment Variables**:

| Name | Value |
|------|-------|
| `NEXT_PUBLIC_API_URL` | `https://your-backend.onrender.com` |

Set it for Production, Preview, and Development environments.

**4. Deploy**

Click **Deploy**. Vercel builds with:
```
npm install --legacy-peer-deps
npm run build
```

Your frontend will be live at `https://your-project.vercel.app`.

**5. Redeploy after backend changes**

When you update `NEXT_PUBLIC_API_URL`, trigger a redeploy:
- Vercel dashboard → Deployments → **Redeploy**

---

## Part 3 — Backend Deployment (Python/FastAPI)

### Option A — Render (Recommended Free Tier)

**1. Create a Render account** at https://render.com

**2. New Web Service**
- Connect GitHub → select `OIBSIP` repo
- **Root Directory:** `DataAnalytics-L2-FraudDetection/app/backend`
- **Environment:** Python 3
- **Build Command:**
  ```
  pip install -r requirements.txt
  ```
- **Start Command:**
  ```
  uvicorn main:app --host 0.0.0.0 --port $PORT
  ```

**3. Add model artifacts**

Render needs the `models/` folder at runtime. Two options:

*Option A-1 — Commit pkl files to git* (simplest, ~100 MB total):
```bash
# Remove models/ from .gitignore temporarily
git add models/*.pkl models/*.json
git commit -m "feat: add trained model artifacts"
git push
```
Render will clone the repo including the models.

*Option A-2 — Render Persistent Disk* ($7/month):
- Render dashboard → Disks → Attach a disk at `/opt/render/project/src/models`
- Upload pkl files via Render shell or deploy script

**4. Update the models/ path**

`app/backend/main.py` resolves models relative to the file:
```python
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
```
This resolves correctly when the repo is cloned in full.

**5. Copy the service URL** into Vercel's `NEXT_PUBLIC_API_URL`.

---

### Option B — Railway

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# From project root
railway init
railway up --service backend \
  --source DataAnalytics-L2-FraudDetection/app/backend
```

Set start command in Railway dashboard:
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

### Option C — Hugging Face Spaces (Free)

1. Create a new Space at https://huggingface.co/spaces
2. Select **Docker** or **Gradio** SDK → choose **Docker**
3. Create a `Dockerfile` in `app/backend/`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY ../../models /app/models
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

4. Push to the Space repo — HF builds and deploys automatically.
5. Set `NEXT_PUBLIC_API_URL=https://your-username-fraud-detection.hf.space`

---

## Part 4 — Environment Variables Reference

### Frontend (`app/frontend/.env.local`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | Yes | `http://localhost:8000` | FastAPI backend base URL |

### Backend (`app/backend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HOST` | No | `0.0.0.0` | Bind address |
| `PORT` | No | `8000` | Listen port (set by host on Render/Railway) |
| `LOG_LEVEL` | No | `info` | uvicorn log level |
| `CORS_ORIGINS` | No | `*` | Comma-separated allowed origins |

> Never commit real `.env` / `.env.local` files. Both are in `.gitignore`.

---

## Part 5 — API Endpoints Reference

Base URL: `http://localhost:8000` (local) or your deployed backend URL.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/api/stats` | Real dataset statistics |
| GET | `/api/metrics` | Real model evaluation metrics |
| GET | `/api/features` | Feature column names (30 features) |
| GET | `/api/feature-importance` | RF importances + LR coefficients |
| GET | `/api/sample-transactions` | 10 real transactions (5 fraud, 5 legit) |
| POST | `/api/predict` | Single transaction prediction |
| POST | `/api/predict/batch` | Batch prediction (up to 100 transactions) |

### POST `/api/predict` — Request Body

```json
{
  "V1": -3.04, "V2": 3.15, "V3": -4.77,
  "V4": 4.36,  "V5": -1.46, "V6": 0.66,
  "V7": -2.77, "V8": 0.21,  "V9": -0.43,
  "V10":-3.26, "V11": 3.05, "V12":-5.07,
  "V13":-1.07, "V14":-7.42, "V15": 0.30,
  "V16":-3.45, "V17":-6.92, "V18":-1.20,
  "V19": 0.51, "V20": 0.28, "V21": 0.67,
  "V22": 0.57, "V23":-0.14, "V24":-0.20,
  "V25": 0.0,  "V26": 0.0,  "V27": 0.13,
  "V28": 0.05, "Amount": 1.00, "Hour": 22,
  "model": "random_forest"
}
```

`model` accepts: `"random_forest"` | `"logistic_regression"` | `"both"`

### POST `/api/predict` — Response

```json
{
  "status": "ok",
  "primary": {
    "prediction": 1,
    "label": "FRAUD",
    "fraud_probability": 0.97,
    "risk_score": 97,
    "risk_level": "CRITICAL",
    "model_used": "Random Forest",
    "disclaimer": "EDUCATIONAL/ANALYTICAL DEMO ONLY..."
  },
  "secondary": null,
  "disclaimer": "EDUCATIONAL/ANALYTICAL DEMO ONLY..."
}
```

---

## Part 6 — Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: imbalanced-learn` | Package not installed | `pip install imbalanced-learn` |
| `FileNotFoundError: creditcard.csv` | Dataset not downloaded | Download from Kaggle and place in `data/` |
| `FileNotFoundError: random_forest.pkl` | Training not run | Run `run_training.bat` / `python run_training.py` |
| Backend CORS error in browser | CORS_ORIGINS too restrictive | Add frontend URL to `CORS_ORIGINS` in backend env |
| Vercel build fails — `next not found` | `npm install` not run | Check `installCommand` in `vercel.json` |
| Render "Application failed to respond" | Wrong start command | Ensure `--port $PORT` in start command |
| `UnicodeEncodeError` on Windows | Windows cp1252 console encoding | Run scripts via `.bat` files or set `PYTHONIOENCODING=utf-8` |
| npm install timeout in OneDrive folder | OneDrive sync locks files | Run `npm install` with OneDrive paused, or move project outside OneDrive |

---

## Part 7 — Production Hardening Checklist

Before using this beyond a demo:

- [ ] Restrict `CORS_ORIGINS` to your specific frontend domain
- [ ] Add API rate limiting (e.g., `slowapi`) to prevent abuse
- [ ] Add authentication (API key or JWT) to prediction endpoints
- [ ] Enable HTTPS-only (handled automatically by Render/Railway/Vercel)
- [ ] Set up monitoring and alerting for backend latency / error rates
- [ ] Add input sanitisation beyond Pydantic validation
- [ ] Log all predictions to an audit trail with timestamps
- [ ] Replace `*` wildcard CORS with explicit domain list
- [ ] Remove `/docs` and `/redoc` endpoints in production builds
- [ ] Test with larger batch sizes and load-test the prediction endpoint

---

*Oasis Infobyte Data Analytics Internship — Level 2, Task 3*
