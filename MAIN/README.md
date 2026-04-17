# AI Student Performance Predictor

Fresh rebuild from the final report.

## MySQL (optional but enabled in app)

Set a MySQL SQLAlchemy URL before launching Streamlit:

```powershell
$env:MYSQL_URL="mysql+pymysql://user:password@host:3306/database_name"
```

Inside the app sidebar:
- paste/test the MySQL URL
- keep `Auto-save upload/manual records + prediction runs` enabled
- run predictions from upload/manual or directly from MySQL records

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional CLI

Train:

```bash
python train.py --input student_data.xlsx
```

Predict (intra-semester, no Final Outcome column required):

```bash
python predict.py --input student_data.xlsx --output predictions.csv
```
