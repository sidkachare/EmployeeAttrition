import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_attrition_distribution(df: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Attrition', data=df)
    plt.title('Attrition Distribution')
    plt.savefig("artifacts/attrition_distribution.png")
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap')
    plt.savefig("artifacts/correlation_heatmap.png")
    plt.close()

def plot_categorical_vs_target(df: pd.DataFrame, categorical_columns: list):
    for col in categorical_columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, hue='Attrition', data=df)
        plt.title(f'{col} vs Attrition')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"artifacts/{col}_vs_attrition.png")
        plt.close()

