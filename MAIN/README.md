# AI Student Performance Predictor

AI-powered system to predict student academic performance based on intra-semester data (CA scores, attendance, etc.). Built with TensorFlow, XGBoost, and scikit-learn.

## 🚀 Quick Start

### Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to HuggingFace Spaces

1. **Create a Space**: Go to [huggingface.co/spaces](https://huggingface.co/spaces) → New Space
2. **Select Streamlit** as the space type
3. **Connect your GitHub repo** and select the MAIN folder as the space folder
4. Space will auto-deploy on every GitHub push

#### Optional: Add MySQL Database (via HF Secrets)

In your HF Space settings, add a secret:
- **Name**: `MYSQL_URL`
- **Value**: `mysql+pymysql://user:password@host:3306/database_name`

Then in the app sidebar, you can enable auto-save for predictions to your database.

---

## 📊 How to Use

1. **Upload Data**: Use the CSV/XLSX upload to import intra-semester student records
2. **View Predictions**: Get instant pass/fail predictions with risk analysis
3. **Export Results**: Download prediction reports as CSV or PDF
4. **Optional DB**: Connect MySQL to auto-save records (see above)

---

## 🏋️ Model Details

Pre-trained ensemble models:
- **LSTM** (TensorFlow): Temporal sequence modeling
- **XGBoost**: Gradient boosting baseline
- **Random Forest**: Ensemble classifier

All models included in `models/` folder.

---

## 💻 CLI Tools (Advanced)

Train new models:
```bash
python train.py --input student_data.xlsx
```

Batch predictions:
```bash
python predict.py --input student_data.xlsx --output predictions.csv
```

---

## 🛠️ Tech Stack

- **Streamlit**: Web interface
- **TensorFlow/Keras**: Neural network models
- **XGBoost, scikit-learn**: Ensemble models
- **Pandas**: Data processing
- **SQLAlchemy/PyMySQL**: Optional database integration
