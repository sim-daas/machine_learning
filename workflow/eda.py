import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('fifa.csv')
df.sample(5)

df.info()
df.describe()

df.isnull().sum()
df.duplicated().sum()

# Select a categorical column
cat_col = 'association'

# Pie chart
plt.figure(figsize=(8, 8))
df[cat_col].value_counts().plot.pie(autopct='%1.1f%%', shadow=True)
plt.title(f'Distribution of {cat_col}')
plt.show()

# Count plot
plt.figure(figsize=(10, 6))
sns.countplot(x=cat_col, data=df)
plt.title(f'Count of {cat_col}')
plt.xticks(rotation=45)
plt.show()




# Bi-variate Analysis
col1 = "association"
col2 = "points"


# Numerical-Numerical

# Scatterplot
sns.scatterplot(x='', y='', data=df)
plt.title("Scatterplot of" + col1 + "and" + col2)
plt.show()

# Line plot (useful for time-series)
sns.lineplot(x='', y='', data=df)
plt.title("lineplot of" + col1 + "and" + col2)
plt.show()


# Numerical - Categorical
# Box plot
sns.boxplot(x=col1, y=col2, data=df)
plt.title('Box plot of' + cat_col)
plt.show()

# Bar Plot
sns.barplot(x=col1, y=col2, data=df)
plt.title("bar plot")
plt.show()

# Dist Plot
sns.kdeplot(data=titanic[titanic['Survived'] == 0], x='Age', label='Not Survived', fill=True)
sns.kdeplot(data=titanic[titanic['Survived'] == 1], x='Age', label='Survived', fill=True)
plt.legend()
plt.title("Age Distribution by Survival Status")
plt.show()


# Category-Category

# Heatmap
heatmap_data = pd.crosstab(df[col1], df[col2])
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='coolwarm')
plt.title(f"Heatmap of {col2} by {col1}")
plt.show()

# Cluster Map
sns.clustermap(heatmap_data, annot=True, cmap='coolwarm')
plt.title(f"Cluster Map of {col2} by {col1}")
plt.show()




























