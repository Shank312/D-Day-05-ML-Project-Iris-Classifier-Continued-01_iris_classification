
# ---------DATA LOADING + INITIAL EXPLORATION---------

# Importing required libraries
import seaborn as sns               # Seaborn is a powerful visualization library based on matplotlib
import pandas as pd                 # Pandas is used for data manipulation and analysis

def load_data():
    # Load the built-in Iris dataset using Seaborn
    df = sns.load_dataset("iris")       # Loads the famous Iris flower dataset as a pandas DataFrame

    # Display the first 5 rows of the dataset to get a quick overview 
    print(df.head())                    # .head() returns the top 5 rows

    # Print the shape (number of rows and columns) of the DataFrame
    print("Shape: ", df.shape)          # Output will be in the form (rows, columns)

    # Display detailed information about the Dataframe
    print(df.info())                    # Includes data types and null checks

    # Check for missing (null) values in each column
    print(df.isnull().sum())            # Ensure data is clean

    return df
