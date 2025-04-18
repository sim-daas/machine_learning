import pandas as pd
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

df['engine'] = df['engine'].str.extract(r'(\d+)').astype(float)
df.head()

# Identify categorical columns (object dtype and integers with few unique values)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = ['brand','fuel_type','accident','clean_title']

# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Remove any categorical columns that might be in numerical format
numerical_cols = [col for col in numerical_cols if col not in categorical_cols]
print(f"\nNumerical columns: {numerical_cols}")
df.dtypes
print(f"\nCategorical columns: {categorical_cols}")

# Create a directory for plots if it doesn't exist
os.makedirs('plots', exist_ok=True)
binary_map_title = {'Yes': 1, 'No': 0}
df['clean_title'] = df['clean_title'].map(binary_map_title)

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

# Create distribution plots for each numerical column
print("\nCreating distribution plots for numerical columns...")
for col in numerical_cols:
    plt.figure(figsize=(12, 6))
    
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot histogram with KDE on first subplot
    sns.histplot(df[col], kde=True, ax=ax1, color='steelblue')
    ax1.set_title(f'Distribution of {col}', fontsize=14)
    ax1.set_xlabel(col, fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    
    # Plot boxplot on second subplot to show outliers
    sns.boxplot(x=df[col], ax=ax2, color='lightgreen')
    ax2.set_title(f'Boxplot of {col}', fontsize=14)
    ax2.set_xlabel(col, fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'plots/{col}_distplot.png')
    plt.show()

print("All distribution plots have been created and saved in the 'plots' directory.")

# Count and print the number of engine values less than 10
engine_lt_10_count = (df['engine'] < 10).sum()
print(f"\nNumber of cars with engine size less than 10: {engine_lt_10_count}")

# Optional: View these records to understand if they might be errors or actual values
if engine_lt_10_count > 0:
    print("\nSample of cars with engine size less than 10:")
    print(df[df['engine'] < 10].head())
