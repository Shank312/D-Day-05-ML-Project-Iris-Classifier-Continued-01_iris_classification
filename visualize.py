
# ---------VISUALIZATION USING SEABORN--------------

import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairplot(df):
    # Pairplot to visualize pairwise relationships in the dataset
    sns.pairplot(df, hue='species')
    plt.suptitle("Pairwise Feature Distribution", y=1.02)  # Add title
    plt.show()

def plot_correlation_heatmap(df):
    # Calculate correlation matrix of numeric features
    corr = df.corr(numeric_only=True)

    # Plot the heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm')         # Annotate correlations
    plt.title("Feature Correlation")                       # Corrected typo: 'tittle' → 'title'
    plt.show()
