
# ------------------ MODEL TRAINING ------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def split_data(df):
    # Separate input features (X) and target labels (Y)
    X = df.drop('species', axis=1)
    Y = df['species']

    # Split into training and testing sets
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def train_models(X_train, Y_train):
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=200)
    lr_model.fit(X_train, Y_train)

    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, Y_train)

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, Y_train)

    return {
        "Logistic Regression": lr_model,
        "KNN": knn_model,
        "Decision Tree": dt_model
    }
