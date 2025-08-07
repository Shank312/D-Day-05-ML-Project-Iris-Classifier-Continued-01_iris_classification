
# ------------------ MODEL EVALUATION ------------------

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_models(models, X_test, Y_test):
    for name, model in models.items():
        Y_pred = model.predict(X_test)

        # Accuracy
        acc = accuracy_score(Y_test, Y_pred)
        print(f"\n{name} Accuracy: {acc:.2f}")

        # Confusion Matrix
        print(confusion_matrix(Y_test, Y_pred))

        # Classification Report
        print(classification_report(Y_test, Y_pred))
