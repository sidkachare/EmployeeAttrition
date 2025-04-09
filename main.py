import os
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data
from src.visualization import (
    plot_attrition_distribution,
    plot_correlation_heatmap,
    plot_categorical_vs_target
)
from src.model_trainer import train_model
from src.model_evaluation import evaluate_model
from src.inference import predict

def main():
    data_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = load_data(data_path)
    os.makedirs("artifacts", exist_ok=True)
    plot_attrition_distribution(df)
    plot_correlation_heatmap(df)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols.remove("Attrition")
    plot_categorical_vs_target(df, cat_cols)

    X_train, X_test, y_train, y_test, scaler, le_dict = preprocess_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    print("\nRunning inference on a few test samples...\n")
    original_df = df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1)
    sample_inputs = original_df.sample(3, random_state=42).drop("Attrition", axis=1)
    predictions, probabilities = predict(sample_inputs)
    for idx, (i, row) in enumerate(sample_inputs.iterrows()):
        print(f"Sample {i}:")
        print(row)
        print(f"Prediction: {'Yes' if predictions[idx]==1 else 'No'} (Attrition)")
        print(f"Confidence: {probabilities[idx][1]*100:.2f}%\n")


if __name__ == "__main__":
    main()
