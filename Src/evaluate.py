import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from preprocessing import load_and_clean_heart

def evaluate_logistic():
    df = load_and_clean_heart()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # classification metrics
    y_pred = model.predict(X_test)
    print("✅ Classification Report")
    print(classification_report(y_test, y_pred))

    # confusion matrix
    print("✅ Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    # ROC curve + AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    score = auc(fpr, tpr)
    print("ROC AUC:", round(score, 3))

    # simple ROC plot
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    evaluate_logistic()
