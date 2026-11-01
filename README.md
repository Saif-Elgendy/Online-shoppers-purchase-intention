# Will This Visitor Buy? — Purchase Intention Prediction

A classical machine-learning project that predicts whether an e-commerce browsing session will result in a purchase.

## Dataset
Online Shoppers Purchasing Intention, UCI Machine Learning Repository (dataset ID 468):
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

The notebook downloads the dataset using `ucimlrepo`. Do not upload the dataset to GitHub; cite the UCI source.

## Project workflow
1. Data understanding and EDA
2. Train/test split
3. Preprocessing in a scikit-learn Pipeline
4. Baseline and multiple classifiers
5. Cross-validation and RandomizedSearchCV
6. Final test evaluation and error analysis
7. Save model with joblib
8. Run the Streamlit application

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Run `01_purchase_intention_ml.ipynb` from top to bottom. It creates:
`artifacts/purchase_intention_pipeline.joblib`

Then run:
```bash
streamlit run app.py
```

## Important
The notebook must be run before the app. All performance numbers in the report, presentation, CV, and LinkedIn post must come from your actual notebook run. Do not claim unmeasured results.
