"""Train models for insurance charges prediction.

Usage:
    python train.py
Saves model to models/ (sklearn joblib)."""


import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROOT = os.path.dirname(os.path.dirname(__file__))

def load_data(path):
    return pd.read_csv(path)

def preprocess(X):
    cat_cols = ['sex','smoker','region']
    num_cols = ['age','bmi','children']
    preproc = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    return preproc

def train():
    data = load_data(os.path.join(ROOT,'insurance.csv'))
    X = data.drop(columns=['charges'])
    y = data['charges']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    preproc = preprocess(X)
    # Linear Regression
    lr = Pipeline([
        ('preproc', preproc),
        ('lr', LinearRegression())
    ])
    lr.fit(X_train, y_train)
    # Random Forest
    rf = Pipeline([
        ('preproc', preproc),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    rf.fit(X_train, y_train)
    # Evaluate
    for name, model in [('LinearRegression', lr), ('RandomForest', rf)]:
        preds = model.predict(X_test)
        print(name)
        print('MAE', mean_absolute_error(y_test, preds))
        print('RMSE', mean_squared_error(y_test, preds, squared=False))
        print('R2', r2_score(y_test, preds))
    # Save best model (choose rf)
    os.makedirs(os.path.join(ROOT,'models'), exist_ok=True)
    joblib.dump(rf, os.path.join(ROOT,'models','rf_model.joblib'))
    print('Saved Random Forest model to models/rf_model.joblib')

if __name__ == '__main__':
    train()
