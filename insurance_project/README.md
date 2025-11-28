
# Medical Insurance Charges Prediction

**Project goal:** Predict medical insurance charges using regression models, provide statistical exploration, and interpret model predictions with SHAP. Includes a Power BI dashboard placeholder and reproducible training scripts.

## Repository structure
```
insurance_project_full/
├── README.md
├── insurance.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── train.py
│   ├── inference.py
│   └── utils.py
├── reports/
│   └── PowerBI_placeholder.txt
├── visuals/
│   ├── distribution_age.png
│   ├── charges_vs_bmi.png
│   └── model_feature_importance.png
├── requirements.txt
└── .gitignore
```

## What I included
- Clean, reproducible Jupyter notebook (`notebooks/analysis.ipynb`) with:
  - Data cleaning and feature engineering
  - Exploratory Data Analysis (plots saved to `visuals/`)
  - Model training: Linear Regression, Random Forest, (optional XGBoost)
  - Evaluation metrics (MAE, RMSE, R²)
  - SHAP interpretability (if SHAP & XGBoost installed; falls back to permutation importances)
- `src/train.py` — script to train and save a model
- `src/inference.py` — load model and run single predictions
- `src/utils.py` — helper functions
- `reports/PowerBI_placeholder.txt` — instructions & a template for building a Power BI dashboard
- `visuals/` — generated PNGs from notebook
- `requirements.txt` — packages used
- `.gitignore` — standard ignores

## How to use
1. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
2. Install requirements: `pip install -r requirements.txt`
3. Run notebook `notebooks/analysis.ipynb` or run training: `python src/train.py`
4. Use `src/inference.py` to load model and predict.

## Notes for resume
- Emphasize regression performance, interpretability using SHAP, and a Power BI dashboard to visualize key drivers of insurance charges.
- Project demonstrates end-to-end ML workflow: data cleaning, EDA, modeling, evaluation, interpretability, and reporting.
