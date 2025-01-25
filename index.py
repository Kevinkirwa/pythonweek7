# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

# Load the Iris dataset
data = load_iris()
iris_df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the species column to the DataFrame
iris_df['species'] = data.target
iris_df['species'] = iris_df['species'].apply(lambda x: data.target_names[x])

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(iris_df.head())

# Check the structure of the dataset
print("\nDataset Info:")
iris_df.info()

# Check for missing values
print("\nMissing Values:")
print(iris_df.isnull().sum())

# Clean the dataset (no missing values in Iris dataset)
# If there were missing values, we could fill or drop them as shown below:
# iris_df.fillna(method='ffill', inplace=True)  # Fill missing values
# iris_df.dropna(inplace=True)  # Drop rows with missing values

# Task 2: Basic Data Analysis

# Compute basic statistics of numerical columns
print("\nBasic Statistics:")
print(iris_df.describe())

# Perform grouping on the 'species' column and compute the mean for each group
species_means = iris_df.groupby('species').mean()
print("\nMean values grouped by species:")
print(species_means)

# Identify interesting patterns (example observation)
print("\nObservation:")
print("Setosa species has the smallest petal length and width, while Virginica has the largest.")

# Task 3: Data Visualization

# Line chart showing trends over columns (example: mean values per species)
species_means.plot(kind='line', figsize=(8, 6))
plt.title("Mean Values of Features by Species")
plt.xlabel("Features")
plt.ylabel("Mean Value")
plt.legend(title="Species")
plt.grid()
plt.show()

# Bar chart showing average petal length per species
sns.barplot(x='species', y='petal length (cm)', data=iris_df, ci=None)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram of petal width
sns.histplot(iris_df['petal width (cm)'], kde=True, bins=10, color='purple')
plt.title("Distribution of Petal Width")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot of sepal length vs petal length
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris_df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# Findings and Observations

# Example findings based on the visualizations and analysis:
print("\nFindings/Observations:")
print("1. Setosa species has the smallest petal length and width, while Virginica has the largest.")
print("2. Petal length and width show a strong correlation in all species, with larger values corresponding to the Virginica species.")
print("3. Sepal length and petal length also have a noticeable positive correlation, with larger sepal length generally corresponding to larger petal length.")
