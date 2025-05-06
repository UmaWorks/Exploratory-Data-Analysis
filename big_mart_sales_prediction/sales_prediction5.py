#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import warnings
warnings.filterwarnings("ignore")


def preprocess_data(train, test, target_transform=True):
    """Advanced preprocessing function with target transformation"""
    print("Starting preprocessing...")
    
    # Store original test for submission
    original_test = test.copy()
    
    # Add dummy target column to test data for consistent processing
    if 'Item_Outlet_Sales' not in test.columns:
        test['Item_Outlet_Sales'] = 0
    
    # Combine datasets for preprocessing
    combined = pd.concat([train, test], ignore_index=True)
    
    # ---- MISSING VALUES HANDLING ----
    
    # Fill missing Item_Weight
    item_weights = combined.groupby('Item_Identifier')['Item_Weight'].mean()
    type_weights = combined.groupby('Item_Type')['Item_Weight'].mean()
    
    # First try to use item-specific weight
    missing_weight = combined['Item_Weight'].isna()
    for idx in combined[missing_weight].index:
        item_id = combined.loc[idx, 'Item_Identifier']
        if item_id in item_weights:
            combined.at[idx, 'Item_Weight'] = item_weights[item_id]
    
    # Then fallback to item type weight
    still_missing = combined['Item_Weight'].isna()
    for idx in combined[still_missing].index:
        item_type = combined.loc[idx, 'Item_Type']
        if item_type in type_weights:
            combined.at[idx, 'Item_Weight'] = type_weights[item_type]
    
    # Final fallback to global mean
    combined['Item_Weight'].fillna(combined['Item_Weight'].mean(), inplace=True)
    
    # ---- OUTLET SIZE HANDLING ----
    
    # Assign specific outlet sizes as requested
    combined.loc[combined['Outlet_Identifier'] == 'OUT010', 'Outlet_Size'] = 'Small'
    combined.loc[combined['Outlet_Identifier'] == 'OUT045', 'Outlet_Size'] = 'Small'
    combined.loc[combined['Outlet_Identifier'] == 'OUT017', 'Outlet_Size'] = 'Small'
    
    # Fill remaining missing Outlet_Size values
    outlet_sizes = combined.groupby('Outlet_Type')['Outlet_Size'].apply(
        lambda x: x.mode()[0] if not x.mode().empty else "Medium"
    )
    
    for idx in combined[combined['Outlet_Size'].isna()].index:
        outlet_type = combined.loc[idx, 'Outlet_Type']
        if outlet_type in outlet_sizes:
            combined.at[idx, 'Outlet_Size'] = outlet_sizes[outlet_type]
        else:
            combined.at[idx, 'Outlet_Size'] = "Medium"  # Default
    
    # ---- FEATURE CLEANING ----
    
    # Standardize Item_Fat_Content as requested
    fat_map = {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}
    combined['Item_Fat_Content'] = combined['Item_Fat_Content'].replace(fat_map)
    
    # Save original Item_Identifier for submission
    combined['Original_Item_Identifier'] = combined['Item_Identifier'].copy()
    
    # Convert Item_Identifier to category names as requested
    combined['Item_Identifier'] = combined['Item_Identifier'].apply(lambda x: x[0:2])
    combined['Item_Identifier'] = combined['Item_Identifier'].map({
        'FD': 'Food', 'NC': 'Non_Consumable', 'DR': 'Drinks'
    })
    
    # Extract Item Category for backward compatibility
    combined['Item_Category'] = combined['Original_Item_Identifier'].str.slice(0, 2)
    combined['Item_Category'] = combined['Item_Category'].map({
        'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'
    })
    
    # Set logical fat content for Non-Consumables
    combined.loc[combined['Item_Category'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Not Applicable'
    
    # ---- HANDLE ZERO VISIBILITY VALUES ----
    
    # Calculate average visibility per item type
    visibility_means = combined[combined['Item_Visibility'] > 0].groupby('Item_Type')['Item_Visibility'].mean()
    global_visibility = combined[combined['Item_Visibility'] > 0]['Item_Visibility'].mean()
    
    # Replace zero visibility with type average
    zero_mask = combined['Item_Visibility'] == 0
    for idx in combined[zero_mask].index:
        item_type = combined.loc[idx, 'Item_Type']
        if item_type in visibility_means:
            combined.at[idx, 'Item_Visibility'] = visibility_means[item_type]
        else:
            combined.at[idx, 'Item_Visibility'] = global_visibility
    
    # ---- FEATURE ENGINEERING ----
    
    # 1. Log transform visibility
    combined['Item_Visibility_Log'] = np.log1p(combined['Item_Visibility'])
    
    # 2. MRP features
    combined['Item_MRP_Squared'] = combined['Item_MRP'] ** 2
    
    # 3. Outlet Years feature (as requested)
    combined['Outlet_Years'] = 2013 - combined['Outlet_Establishment_Year']
    
    # Create Era categorical feature (preserved from original)
    combined['Establishment_Era'] = pd.cut(
        combined['Outlet_Establishment_Year'],
        bins=[1980, 1990, 2000, 2010, 2013],
        labels=['Very Old', 'Old', 'Recent', 'New']
    )
    
    # 4. Price and weight features
    combined['Price_Per_Weight'] = combined['Item_MRP'] / combined['Item_Weight']
    combined['Visibility_To_MRP'] = combined['Item_Visibility'] / combined['Item_MRP']
    
    # 5. Interaction features
    combined['Item_Type_X_Outlet_Type'] = combined['Item_Type'] + '_' + combined['Outlet_Type']
    combined['Item_Category_X_Outlet_Type'] = combined['Item_Category'] + '_' + combined['Outlet_Type']
    
    # 6. Tree-specific features
    combined['MRP_X_Weight'] = combined['Item_MRP'] * combined['Item_Weight']
    combined['MRP_X_Visibility'] = combined['Item_MRP'] * combined['Item_Visibility']
    combined['MRP_X_Outlet_Years'] = combined['Item_MRP'] * combined['Outlet_Years']
    
    # ---- ENCODING CATEGORICAL VARIABLES ----
    
    # List of categorical columns to encode (added Item_Identifier)
    cat_cols = [
        'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
        'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type',
        'Item_Category', 'Establishment_Era', 'Item_Identifier',
        'Item_Type_X_Outlet_Type', 'Item_Category_X_Outlet_Type'
    ]
    
    # Apply one-hot encoding
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=False)
    
    # ---- DROP UNNECESSARY COLUMNS ----
    # Drop Original_Item_Identifier and Outlet_Establishment_Year as requested
    cols_to_drop = ['Original_Item_Identifier', 'Outlet_Establishment_Year']
    combined.drop(cols_to_drop, axis=1, inplace=True)
    
    # ---- TRANSFORM TARGET VARIABLE ----
    target_transformer = None
    if target_transform:
        print("Applying target transformation...")
        # Get train part of combined data for target transformation
        train_mask = combined.index < len(train)
        train_y = combined.loc[train_mask, 'Item_Outlet_Sales']
        
        # Apply power transformation to target
        target_transformer = PowerTransformer(method='yeo-johnson')
        transformed_y = target_transformer.fit_transform(train_y.values.reshape(-1, 1)).flatten()
        
        # Update the target in the combined data
        combined.loc[train_mask, 'Item_Outlet_Sales'] = transformed_y
    
    # Split back into train and test
    train_processed = combined.iloc[:len(train)]
    test_processed = combined.iloc[len(train):]
    
    # Remove target column from test
    X_test = test_processed.drop('Item_Outlet_Sales', axis=1)
    y_train = train_processed['Item_Outlet_Sales']
    X_train = train_processed.drop('Item_Outlet_Sales', axis=1)
    
    print(f"Preprocessing complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, original_test, target_transformer


def train_weighted_ensemble(X_train, y_train, cv=5):
    """Train multiple models and create weighted ensemble"""
    print("Training ensemble models...")
    
    # Create model dictionary
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1,
            objective='reg:squarederror',
            random_state=42
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=5,
            min_child_samples=5,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.03,
            reg_lambda=0.3,
            min_split_gain=0.01,
            min_data_in_leaf=10,
            random_state=42,
            verbose=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
    }
    
    # Train models
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    # Calculate cross-validation RMSE for each model
    print("Calculating cross-validation performance...")
    cv_errors = {}
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for name, model in models.items():
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            # Split data
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train and predict
            model_clone = clone(model)
            model_clone.fit(X_train_fold, y_train_fold)
            preds = model_clone.predict(X_val_fold)
            
            # Calculate error
            rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
            scores.append(rmse)
        
        cv_errors[name] = np.mean(scores)
        print(f"  {name} CV RMSE: {cv_errors[name]:.4f}")
    
    # Calculate weights based on inverse squared error
    inverse_squared_errors = {name: 1/(error**2) for name, error in cv_errors.items()}
    sum_weights = sum(inverse_squared_errors.values())
    weights = {name: w/sum_weights for name, w in inverse_squared_errors.items()}
    
    print("Final model weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    
    return trained_models, weights

# ==============================================
# PREDICTION FUNCTION
# ==============================================
def predict_with_ensemble(trained_models, weights, X_test, target_transformer=None):
    """Generate weighted ensemble predictions"""
    print("Generating ensemble predictions...")
    
    # Make predictions with each model
    all_preds = {}
    for name, model in trained_models.items():
        all_preds[name] = model.predict(X_test)
    
    # Create weighted average prediction
    ensemble_pred = np.zeros(X_test.shape[0])
    for name, preds in all_preds.items():
        ensemble_pred += weights[name] * preds
    
    # Inverse transform if target was transformed
    if target_transformer is not None:
        ensemble_pred = target_transformer.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
    
    # Ensure all predictions are non-negative
    return np.maximum(ensemble_pred, 0)


def analyze_feature_importance(model, X_train):
    """Analyze and plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        top_features = feature_importance.head(20)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 20 Features by Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        return feature_importance
    else:
        print("Model doesn't have feature_importances_ attribute")
        return None


def optimized_sales_prediction(train_path, test_path):
    """Complete pipeline for BigMart sales prediction"""
    # Load data
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Preprocess data with target transformation
    X_train, y_train, X_test, original_test, target_transformer = preprocess_data(
        train, test, target_transform=True
    )
    
    # Train weighted ensemble
    trained_models, weights = train_weighted_ensemble(X_train, y_train, cv=5)
    
    # Generate predictions
    predictions = predict_with_ensemble(trained_models, weights, X_test, target_transformer)
    
    # Create submission file
    submission = pd.DataFrame({
        'Item_Identifier': original_test['Item_Identifier'],
        'Outlet_Identifier': original_test['Outlet_Identifier'],
        'Item_Outlet_Sales': predictions
    })
    
    # Save submission
    submission.to_csv('optimized_ensemble_submission.csv', index=False)
    print("\nSubmission file saved as 'optimized_ensemble_submission.csv'")
    
    # Analyze feature importance using the model with highest weight
    best_model_name = max(weights.items(), key=lambda x: x[1])[0]
    best_model = trained_models[best_model_name]
    
    print(f"\nAnalyzing feature importance using {best_model_name}...")
    feature_importance = analyze_feature_importance(best_model, X_train)
    
    if feature_importance is not None:
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
    
    return submission



if __name__ == "__main__":
    submission = optimized_sales_prediction(
        "train_v9rqX0R.csv", 
        "test_AbJTz2l.csv"
    )