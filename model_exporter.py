
# --------- MODEL EXPORTER WITH COMMENTS ---------

# Import necessary libraries
import joblib  # Used to save and load machine learning models in .pkl format
import os      # Provides functions to interact with the operating system (like file paths)

def save_model(model, filename="iris_decision_tree_model.pkl"):
    """
    Saves the trained machine learning model to a .pkl file using joblib.

    Parameters:
    - model: The trained model object you want to save (e.g., DecisionTreeClassifier)
    - filename: The name of the output file (default is 'iris_decision_tree_model.pkl')
    """

    # Get the absolute path of the filename (e.g., convert to: D:/.../iris_decision_tree_model.pkl)
    path = os.path.abspath(filename)

    # Save the model to the given path using joblib (efficient for large NumPy arrays)
    joblib.dump(model, path)

    # Print the full path where the model has been saved
    print(f"Model saved at: {path}")

