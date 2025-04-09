from EDAPro import module
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import sklearn
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import skimpy as skim


# Configuration
pd.set_option("display.max_columns", None)
sklearn.set_config(transform_output="pandas")
warnings.filterwarnings("ignore")

# Data Loading
def load_data(file_path):
    return pd.read_csv(file_path, index_col=0)

# EDA Functions
def perform_eda(df):
    print("Sample data:")
    display(df.sample(5))
    
    print("\nDataset Info:")
    display(df.info())
    
    print("\nDescriptive Statistics:")
    display(df.describe())
    
    print("\nData Types:")
    display(df.dtypes)
    
    # Correlation analysis
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(22, 12))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()
    
    # Descriptive statistics for selected columns
    display(df.loc[:, ['Appliances', 'lights', 'hour']].describe())
    
    # Distribution of lights
    sns.displot(data=df, x='lights', kde=True)
    plt.title('Light Distribution')
    plt.show()
    
    return df

# Feature Engineering
def engineer_features(df):
    return (df
        .assign(
            date = lambda x: pd.to_datetime(x['date'], dayfirst=True),
            hour = lambda x: x['date'].dt.hour,
            min = lambda x: x['date'].dt.minute
        )
    )

# Outlier Removal
def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column in the DataFrame using pandas functions
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing the data
    column : str
        The column name to check for outliers
    method : str, optional (default='iqr')
        The method to use for outlier detection ('iqr' or 'zscore')
    threshold : float, optional (default=1.5)
        The threshold value for the IQR method
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with outliers removed
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    original_size = len(df_copy)
    
    # Visualize original distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df_copy[column], kde=True)
    plt.title(f'Distribution of {column} Before Outlier Removal')
    
    if method == 'iqr':
        # Calculate quartiles and IQR
        q1 = df_copy[column].quantile(0.25)
        q3 = df_copy[column].quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # Filter out outliers
        df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
        
        print(f"IQR method: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
        print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
        
    elif method == 'zscore':
        # Calculate mean and standard deviation
        mean = df_copy[column].mean()
        std = df_copy[column].std()
        
        # Define bounds using z-score
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        # Filter out outliers
        df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
        
        print(f"Z-score method: Mean={mean:.2f}, Std={std:.2f}")
        print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
        
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    # Print number of removed outliers
    removed = original_size - len(df_copy)
    print(f"Removed {removed} outliers ({removed/original_size:.2%} of data) from {column} column")
    
    # Visualize cleaned distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df_copy[column], kde=True)
    plt.title(f'Distribution of {column} After Outlier Removal')
    
    plt.tight_layout()
    plt.show()
    
    return df_copy

# Data Preprocessing
def preprocess_data(df, keep_cols=['Appliances', 'hour', 'lights']):
    # Select relevant columns
    df_selected = df.loc[:, keep_cols]
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Scale features (excluding target)
    features = keep_cols[1:]  # All columns except 'Appliances'
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_selected.loc[:, features]),
        columns=features,
        index=df_selected.index
    )
    
    # Add target variable
    df_scaled = df_scaled.join(df_selected['Appliances'])
    
    return df_scaled, scaler, features

# Model Training
def train_model(df_scaled):
    # Split data
    X = df_scaled.drop(columns=['Appliances'])
    y = df_scaled['Appliances']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return rf_model

# Generate Predictions
def generate_predictions(test_df, scaler, features, model):
    # Process test data the same way as training data
    test_df = engineer_features(test_df)
    
    # Select features
    test_features = test_df[features]
    
    # Scale features
    test_scaled = pd.DataFrame(
        scaler.transform(test_features),
        columns=features,
        index=test_features.index
    )
    
    # Make predictions
    predictions = model.predict(test_scaled)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'Appliances': predictions
    }, index=test_df.index)
    
    return submission

# Main execution flow
if __name__ == "__main__":
    # Load train data
    train_df = load_data('train.csv')
    skim.skim(train_df)
    module.missing_info(train_df)
    # Engineer features
    train_df = engineer_features(train_df)
    
    # Remove outliers from Appliances column (before scaling)
    train_df = remove_outliers(train_df, 'Appliances', method='iqr', threshold=1.5)
    
    # Preprocess data
    keep_cols = ['Appliances', 'hour', 'lights']
    df_scaled, scaler, features = preprocess_data(train_df, keep_cols)
    
    # Train model
    model = train_model(df_scaled)
    
    # Load test data and generate predictions
    test_df = load_data('test.csv')
    submission = generate_predictions(test_df, scaler, features, model)
    
    # Save submission
    submission.to_csv('submission.csv')
    print("Submission file created successfully!")

















