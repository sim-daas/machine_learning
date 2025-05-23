import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import skimpy as skim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import math
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor






def clean_car_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the car dataset without using .apply() for these steps."""
    # Make a copy to avoid modifying the original DataFrame if passed by reference
    df = df.copy()
    df = df.drop(columns=['int_col', 'ext_col'])

    # Strip whitespace from object columns using vectorized .str accessor
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # Drop duplicate rows using the built-in method
    df.drop_duplicates(inplace=True)

    # Handle missing fuel_type based on engine description using boolean indexing and .loc
    index_to_drop = df[df.fuel_type.isna() & ~df.engine.str.contains('Electric', na=False)].index
    df.drop(index_to_drop, inplace=True)
    df.loc[df.engine.str.contains('Electric', na=False), 'fuel_type'] = 'Electric'

    # Drop rows with missing 'accident' information using dropna
    df.dropna(subset=['accident'], inplace=True)

    # Fill missing 'clean_title' with 'No' using fillna or .loc
    df['clean_title'].fillna('No', inplace=True)
    df.loc[df.accident.str.contains('1'), 'clean_title'] = 'Yes'
    # df.loc[df.clean_title.isna(), 'clean_title'] = 'No' # Alternative

    return df

def encode_car_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical features of the car dataset."""
    df_encoded = df.copy()
    print(df_encoded.head())
    # Label Encoding for binary-like columns
    # Adjust mapping based on actual values observed after cleaning
    binary_map_accident = {'At least 1 accident': 1, 'None reported': 0}
    binary_map_title = {'Yes': 1, 'No': 0}

    df_encoded['accident'] = df_encoded['accident'].map(binary_map_accident)
    df_encoded['clean_title'] = df_encoded['clean_title'].map(binary_map_title)

    # One-Hot Encoding for specified columns
    cols_to_encode = ['brand', 'fuel_type']
    df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode, drop_first=True, dummy_na=False)
    print(df_encoded.head())
    return df_encoded

def train(df_encoded: pd.DataFrame, feature_cols: list, model, test_size=0.2, random_state=42):
    """
    Trains a model on the encoded data and evaluates its performance.
    
    Args:
        df_encoded: Cleaned and encoded DataFrame
        feature_cols: List of column names to use as features
        model: Scikit-learn model object with fit and predict methods
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, rmse, r2_score)
    """
    # Prepare features and target
    X = df_encoded[feature_cols]
    y = df_encoded['price']
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Optional: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train the model
    print(f"Training {type(model).__name__} model...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print(f"Model performance:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Return the trained model and metrics
    return {
        'model': model,
        'scaler': scaler,
        'rmse': rmse,
        'r2': r2,
        'feature_cols': feature_cols
    }

def predict_and_submit(model_results, test_path='test.csv', submission_path='submission.csv'):
    """
    Makes predictions on test data and creates a submission file.
    
    Args:
        model_results: Dictionary returned from train function containing model and scaler
        test_path: Path to the test CSV file
        submission_path: Path to save the submission CSV file
    """
    # Load test data
    test_df = pd.read_csv(test_path, index_col=0)
    
    # Clean and encode test data using the same functions
    #test_df_cleaned = clean_car_data(test_df)
    test_df_encoded = encode_car_data(test_df)
    
    # Extract components from model_results
    model = model_results['model']
    scaler = model_results['scaler']
    feature_cols = model_results['feature_cols']
    
    # Ensure all required columns exist in test data
    for col in feature_cols:
        if col not in test_df_encoded.columns:
            test_df_encoded[col] = 0
    
    # Select only the required features and in the correct order
    X_test = test_df_encoded[feature_cols]
    
    # Scale features using the same scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'price': predictions
    }, index=test_df_encoded.index)
    
    # Save to CSV
    submission.to_csv(submission_path)
    print(f"Submission file created at {submission_path}")
    return submission

df = pd.read_csv('train.csv', index_col=0)
df_cleaned = clean_car_data(df)
df_encoded = encode_car_data(df_cleaned)
df_encoded.isna().sum()

df_encoded.head()
df_encoded.columns

data = train(df_encoded=df_encoded,
             feature_cols=['model_year', 'milage', 'accident', 'clean_title'] + [col for col in df_encoded.columns if col.startswith('fuel_type_') or col.startswith('brand_')],
             model=CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42, verbose=0),
             test_size=0.2,
             random_state=42)

# Extract and print values from the training results
print(f"Trained Model: {data['model']}")
print(f"Scaler: {data['scaler']}")
print(f"RMSE: ${data['rmse']:.2f}")
print(f"R² Score: {data['r2']:.4f}")
print(f"Features used: {data['feature_cols']}")

predict_and_submit(data,
                   test_path='test.csv',
                   submission_path='submission.csv')

# Add model comparison loop to evaluate multiple models
print("\n=== COMPARING MULTIPLE REGRESSION MODELS ===\n")

# Dictionary of models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
    'CatBoost': CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42, verbose=0)
}

# Feature columns to use
feature_cols = ['model_year', 'milage', 'accident', 'clean_title'] + [col for col in df_encoded.columns if col.startswith('fuel_type_')]

# Store results for comparison
results = {}

# Loop through each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    try:
        # Train the model and get results
        model_results = train(
            df_encoded=df_encoded,
            feature_cols=feature_cols,
            model=model,
            test_size=0.2,
            random_state=42
        )
        
        # Store results
        results[model_name] = {
            'rmse': model_results['rmse'],
            'r2': model_results['r2'],
            'model': model_results['model'],
            'scaler': model_results['scaler']
        }
        
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")

# Find the best model based on RMSE
if results:
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    print("\n=== BEST MODEL ===")
    print(f"Model: {best_model[0]}")
    print(f"RMSE: ${best_model[1]['rmse']:.2f}")
    print(f"R² Score: {best_model[1]['r2']:.4f}")
    
    # Generate submission with the best model
    print("\nGenerating submission with the best model...")
best_model_results = {
        'model': best_model[1]['model'],
        'scaler': best_model[1]['scaler'],
        'feature_cols': feature_cols
    }
predict_and_submit(
        model_results=best_model_results,
        test_path='test.csv',
        submission_path='submission.csv'
    )

# Generate correlation plot for numerical columns
print("\n=== CORRELATION ANALYSIS OF NUMERICAL FEATURES ===\n")
df_encoded.accident.dtypes
# Select only numerical columns
numeric_columns = df_encoded.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df_encoded[numeric_columns].corr()

# Create a correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')  # Save the figure to a file
plt.show()  # Display the plot if running in an interactive environment

# Print the correlation with target variable (price)
if 'price' in numeric_columns:
    price_correlation = correlation_matrix['price'].sort_values(ascending=False)
    print("\nCorrelation with Price (Target Variable):")
    print(price_correlation)








