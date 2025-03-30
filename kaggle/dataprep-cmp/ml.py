import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import sklearn
import warnings


pd.set_option("display.max_columns", None)
sklearn.set_config(transform_output="pandas")
warnings.filterwarnings("ignore")


df = pd.read_csv('train.csv', index_col=0)
df.sample(5)

df.info()
df.describe()
corr = df.corr(numeric_only=True)
df.dtypes


df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['hour'] = df['date'].dt.hour
df['min'] = df['date'].dt.minute
df.drop(columns=['date', 'rv1', 'rv2', 'Visibility', 'Tdewpoint'])


plt.figure(figsize=(22, 12))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

df.loc[:, ['Appliances', 'lights','T2', 'T3','RH_5','RH_out','hour']].describe()


sns.displot(data=df, x='lights', kde=True )
plt.title('light distplot')
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

keep_cols = ['Appliances', 'lights', 'T2', 'T3', 'RH_5', 'RH_out', 'hour']

df = df.loc[:, keep_cols]

df_scaled = pd.DataFrame(
    scaler.fit_transform(df.loc[:, keep_cols[1:]]),
    columns=df.columns[1:],
    index=df.index
)

df_scaled = df_scaled.join(df['Appliances'])
df_scaled.head()

# Add RandomForestRegressor model with 80/20 train-test split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Split the data into features (X) and target (y)
X = df_scaled.drop(columns=['Appliances'])
y = df_scaled['Appliances']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestRegressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Display feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Load test data for submission
test_df = pd.read_csv('test.csv', index_col=0)

# Process the date field the same way as training data
test_df['date'] = pd.to_datetime(test_df['date'], dayfirst=True)
test_df['hour'] = test_df['date'].dt.hour
test_df['min'] = test_df['date'].dt.minute

# Select the same columns as used in training
test_features = test_df[keep_cols[1:]]  # Exclude 'Appliances' which is our target

# Scale the test features using the same scaler
test_scaled = pd.DataFrame(
    scaler.transform(test_features),
    columns=test_features.columns,
    index=test_features.index
)

# Make predictions
test_predictions = rf_model.predict(test_scaled)

# Create submission DataFrame
submission = pd.DataFrame({
    'Appliances': test_predictions
}, index=test_df.index)

# Save to CSV
submission.to_csv('submission.csv')
print("Submission file created successfully!")

















