import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=False)
    print("Classification Report:\n", report)

    with open("artifacts/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()
