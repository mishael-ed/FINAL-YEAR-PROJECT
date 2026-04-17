# Deploy to HuggingFace Spaces (Free Tier)

Follow these steps to deploy your Student Academic Predictor to HuggingFace Spaces for free.

## Prerequisites

- GitHub account (repo with this code)
- HuggingFace account (free)
- The MAIN folder from your repo

## Step 1: Push to GitHub

Ensure your code is on GitHub. The repo should look like:

```
MAIN/
├── app.py
├── requirements.txt
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── models/
│   ├── lstm_model.keras
│   ├── lstm_scaler.joblib
│   ├── rf_model.joblib
│   └── xgboost_model.joblib
└── sap/
    ├── __init__.py
    ├── db.py
    ├── features.py
    ├── io.py
    ├── model.py
    └── reporting.py
```

## Step 2: Create HuggingFace Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in details:
   - **Space name**: `student-predictor` (or your preferred name)
   - **Space type**: `Streamlit`
   - **Visibility**: `Public` (free tier)
4. Click **"Create Space"**

## Step 3: Connect GitHub Repository

1. In your Space settings, scroll to **"Repository"**
2. Click **"Connect to GitHub"** (or use Git CLI)
3. Select your repo branch and point to the **MAIN folder**
4. Save settings

## Step 4: Auto-Deploy

Once connected, HF Spaces will:
- ✅ Auto-install dependencies from `requirements.txt`
- ✅ Auto-start `app.py` 
- ✅ Auto-redeploy on every GitHub push

Your app will be live at: `https://huggingface.co/spaces/[your-username]/student-predictor`

## Step 5: (Optional) Add MySQL Secret

To enable database features:

1. In Space settings → **"Repository"** → **"Secrets"**
2. Add a new secret:
   - **Name**: `MYSQL_URL`
   - **Value**: `mysql+pymysql://user:password@host:3306/database`
   - Click **"Add secret"**

3. The app will automatically use this MySQL connection if available

## Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"
- Check `requirements.txt` has all dependencies
- Restart the Space (Settings → Restart Space)

### "Streamlit cloud is running out of memory"
- TensorFlow is large; HF Spaces typically handles it fine
- If you hit memory limits, contact HF support for GPU allocation

### Models not loading
- Ensure `models/` folder is in GitHub repo
- Check file paths in `sap/model.py` use relative paths

## Monitoring

- **Logs**: Click "Logs" in Space settings to debug issues
- **Activity**: See deployment history in "Activity" tab

## Free Tier Limits

| Feature | Limit |
|---------|-------|
| Number of Spaces | Unlimited |
| Storage | 50GB per Space |
| Compute | CPU-only (no GPU) |
| Runtime | Persistent (always on) |
| Bandwidth | Unlimited |

Your app runs 24/7 on HF Spaces free tier! 🎉
