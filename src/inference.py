import joblib
import numpy as np
import pandas as pd

def load_artifacts():
    model = joblib.load("artifacts/xgb_model.joblib")
    scaler = joblib.load("artifacts/scaler.joblib")
    label_encoders = joblib.load("artifacts/label_encoders.joblib")
    return model, scaler, label_encoders

def preprocess_input(input_df: pd.DataFrame, label_encoders: dict, scaler):
    df = input_df.copy()

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    drop_cols = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    scaled_input = scaler.transform(df)
    return scaled_input

def predict(input_df: pd.DataFrame):
    model, scaler, label_encoders = load_artifacts()
    processed_input = preprocess_input(input_df, label_encoders, scaler)
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)
    return prediction, prediction_proba
