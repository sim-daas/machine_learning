import pandas as pdV
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set plot style and figure size
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
df = pd.read_csv('train.csv')

# Print basic information about the dataset
print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDescriptive statistics:")
print(df.describe())

# Identify categorical columns (object dtype and integers with few unique values)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Add integer columns with few unique values (typically < 10) to categorical columns
for col in df.select_dtypes(include=['int64', 'int32']).columns:
    if df[col].nunique() < 10:
        categorical_cols.append(col)

print(f"\nCategorical columns: {categorical_cols}")

# Create a directory for plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Create countplots for each categorical column
for col in categorical_cols:
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, x=col, palette='viridis')
    
    # Improve readability for plots with many categories
    if df[col].nunique() > 10:
        plt.xticks(rotation=90)
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)
    
    plt.title(f'Distribution of {col}', fontsize=15)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'plots/{col}_countplot.png')
    plt.show()

print("All countplots have been created and saved in the 'plots' directory.")
