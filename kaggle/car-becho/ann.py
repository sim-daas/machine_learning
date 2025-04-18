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
# Add TensorFlow and Keras imports
from tensorflow import keras
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.optimizers import adam_v2


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

def build_neural_network(df_encoded: pd.DataFrame, feature_cols: list, test_size=0.2, random_state=42,
                         epochs=10, batch_size=32, patience=10):
    """
    Builds and trains a neural network for predicting car prices.
    
    Args:
        df_encoded: Cleaned and encoded DataFrame
        feature_cols: List of column names to use as features
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        patience: Number of epochs with no improvement after which training will be stopped
        
    Returns:
        Dictionary containing the trained model, scaler, and performance metrics
    """
    # Prepare features and target
    X = df_encoded[feature_cols]
    y = df_encoded['price']
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Get input dimensions
    input_dim = X_train_scaled.shape[1]
    
    # Build the neural network
    print("Building neural network model...")
    model = Sequential([
        # Input layer
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        
        # Hidden layers
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        
        # Output layer - single neuron for regression
        Dense(1)
    ])
    
    # Compile the model
    model.compile(
        optimizer=adam_v2.Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )
    
    # Create callbacks for early stopping and model checkpointing
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint('best_nn_model.h5', save_best_only=True, monitor='val_loss')
    ]
    
    # Train the model
    print("Training neural network...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Make predictions
    y_pred = model.predict(X_val_scaled).flatten()
    
    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print(f"Neural Network performance:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Predictions vs Actual')
    
    plt.tight_layout()
    plt.savefig('neural_network_results.png')
    plt.show()
    
    # Return the model and metrics in the same format as the train function
    return {
        'model': model,
        'scaler': scaler,
        'rmse': rmse,
        'r2': r2,
        'feature_cols': feature_cols,
        'history': history.history
    }

def predict_with_nn(model_results, test_path='test.csv', submission_path='nn_submission.csv'):
    """
    Makes predictions using the neural network model and creates a submission file.
    
    Args:
        model_results: Dictionary returned from build_neural_network function
        test_path: Path to the test CSV file
        submission_path: Path to save the submission CSV file
    """
    # Load test data
    test_df = pd.read_csv(test_path, index_col=0)
    
    # Clean and encode test data
    test_df_cleaned = clean_car_data(test_df)
    test_df_encoded = encode_car_data(test_df_cleaned)
    
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
    predictions = model.predict(X_test_scaled).flatten()
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'price': predictions
    }, index=test_df_encoded.index)
    
    # Save to CSV
    submission.to_csv(submission_path)
    print(f"Neural network submission file created at {submission_path}")
    return submission

# Example usage at the end of the file
if __name__ == "__main__":
    # Load and prepare data
    df = pd.read_csv('train.csv', index_col=0)
    df_cleaned = clean_car_data(df)
    df_encoded = encode_car_data(df_cleaned)
    
    # Select features
    feature_cols = ['model_year', 'milage', 'accident', 'clean_title'] + [col for col in df_encoded.columns if col.startswith('brand_') or col.startswith('fuel_type_')]
    
    # Train neural network
    nn_results = build_neural_network(
        df_encoded=df_encoded,
        feature_cols=feature_cols,
        epochs=10,
        batch_size=32
    )
    
    # Create submission file
    predict_with_nn(nn_results, test_path='test.csv', submission_path='neural_network_submission.csv')