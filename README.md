# 🌸 Iris Classifier – Decision Tree Model (Day 05)

This project is a continuation of the Iris flower classification task using the **Decision Tree Classifier**. It demonstrates the full ML workflow including:

- Data loading & preprocessing
- Exploratory Data Analysis (EDA)
- Model training
- Model evaluation
- Saving & loading the trained model (`.pkl`)
- Final prediction using `main.py`

---

## 📂 Project Structure

01_iris_classification/
├── data_loader.py # Loads and inspects the Iris dataset
├── visualize.py # Visualizes feature distributions (optional)
├── model_trainer.py # Trains the Decision Tree model
├── evaluator.py # Evaluates model accuracy
├── model_exporter.py # Saves the trained model (.pkl)
├── iris_decision_tree_model.pkl # Trained model file
├── main.py # Runs the complete pipeline
├── pycache/ # Python cache files (auto-generated)


---

## 🧠 Dataset Info

- **Source**: `sklearn.datasets.load_iris()`
- **Samples**: 150
- **Features**:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- **Target**: Species (Setosa, Versicolor, Virginica)

---

## 🚀 How to Run the Project

### ▶️ 1. Clone the Repository

```bash
git clone https://github.com/Shank312/D-Day-05-ML-Project-Iris-Classifier-Continued-01_iris_classification.git
cd D-Day-05-ML-Project-Iris-Classifier-Continued-01_iris_classification

▶️ 2. Install Required Libraries
Create a virtual environment (recommended):
pip install -r requirements.txt
If requirements.txt is missing, install manually:
pip install pandas scikit-learn joblib matplotlib seaborn

▶️ 3. Run the Project
python main.py

This will:

Load and display the dataset

Train a Decision Tree classifier

Evaluate and print accuracy

Save the model to iris_decision_tree_model.pkl


🧪 Sample Output
   sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
...
Shape: (150, 5)
<class 'pandas.core.frame.DataFrame'>
...
Model saved at: D:/.../iris_decision_tree_model.pkl
Accuracy: 0.97

💾 Model Saving/Loading
Save model:
from model_exporter import save_model
save_model(model)

Load model:
from joblib import load
model = load("iris_decision_tree_model.pkl")

📈 Future Improvements
Try different classifiers: KNN, SVM, RandomForest
Export as a Flask API
Add Jupyter Notebook version
Automate hyperparameter tuning

👨‍💻 Author
Shankar Kumar
GitHub: @Shank312
Project: Day 05 – Machine Learning Iris Classifier (Continued)

📄 License
This project is open-source and available under the MIT License.

