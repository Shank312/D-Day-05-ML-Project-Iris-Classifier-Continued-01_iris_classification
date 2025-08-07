
# ----------------- MAIN SCRIPT -----------------

from data_loader import load_data
from visualize import plot_pairplot, plot_correlation_heatmap
from model_trainer import split_data, train_models
from evaluator import evaluate_models
from model_exporter import save_model

def main():
    # Step 1: Load and explore data
    df = load_data()

    # Step 2: Visualizations
    plot_pairplot(df)
    plot_correlation_heatmap(df)

    # Step 3: Train-test split
    X_train, X_test, Y_train, Y_test = split_data(df)

    # Step 4: Train classifiers
    models = train_models(X_train, Y_train)

    # Step 5: Evaluate models
    evaluate_models(models, X_test, Y_test)

    # Step 6: Export the best model (assuming Decision Tree)
    best_model = models["Decision Tree"]
    save_model(best_model)

if __name__ == "__main__":
    main()
