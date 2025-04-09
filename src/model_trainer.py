import xgboost as xgb
from sklearn.metrics import classification_report
import joblib
import os

def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/xgb_model.joblib")

    return model
