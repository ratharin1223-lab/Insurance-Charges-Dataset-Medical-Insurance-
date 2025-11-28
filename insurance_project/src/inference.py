"""Load saved model and run a single prediction example."""
import os, joblib, pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, 'models', 'rf_model.joblib')

def predict(example_dict):
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([example_dict])
    preds = model.predict(df)
    return preds[0]

if __name__ == '__main__':
    example = {'age':33, 'sex':'male', 'bmi':22.0, 'children':1, 'smoker':'no', 'region':'southwest'}
    print('Prediction for example:', predict(example))
