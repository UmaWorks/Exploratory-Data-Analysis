#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


def preprocess_data(train, test):
    train_processed = train.copy()
    test_processed = test.copy()

    original_test = test_processed.copy()
    
    test_processed['Item_Outlet_Sales'] = 0 
    combined = pd.concat([train_processed, test_processed])
    
    # Fix Item_Weight missing values using mean of same Item_Identifier
    item_weights = {}
    for item_id in combined['Item_Identifier'].unique():
        item_mask = combined['Item_Identifier'] == item_id
        weights = combined.loc[item_mask, 'Item_Weight'].dropna()
        if len(weights) > 0:
            item_weights[item_id] = weights.mean()
    
    for idx, row in combined[combined['Item_Weight'].isna()].iterrows():
        item_id = row['Item_Identifier']
        if item_id in item_weights:
            combined.at[idx, 'Item_Weight'] = item_weights[item_id]
    
    # If still missing, use mean of same Item_Type
    type_weights = {}
    for item_type in combined['Item_Type'].unique():
        type_mask = combined['Item_Type'] == item_type
        weights = combined.loc[type_mask, 'Item_Weight'].dropna()
        if len(weights) > 0:
            type_weights[item_type] = weights.mean()
    
    for idx, row in combined[combined['Item_Weight'].isna()].iterrows():
        item_type = row['Item_Type']
        if item_type in type_weights:
            combined.at[idx, 'Item_Weight'] = type_weights[item_type]
    
    # Fill any remaining missing weights with overall mean
    combined['Item_Weight'].fillna(combined['Item_Weight'].mean(), inplace=True)
    
    # Fill missing Outlet_Size values based on Outlet_Type
    outlet_sizes = {}
    for outlet_type in combined['Outlet_Type'].unique():
        outlet_mask = combined['Outlet_Type'] == outlet_type
        sizes = combined.loc[outlet_mask, 'Outlet_Size'].dropna()
        if len(sizes) > 0:
            outlet_sizes[outlet_type] = sizes.mode()[0]
    
    for idx, row in combined[combined['Outlet_Size'].isna()].iterrows():
        outlet_type = row['Outlet_Type']
        if outlet_type in outlet_sizes:
            combined.at[idx, 'Outlet_Size'] = outlet_sizes[outlet_type]
    
    # Fill missing Outlet_Size based on specific Outlet_Identifier values
    combined.loc[combined['Outlet_Identifier'] == 'OUT010', 'Outlet_Size'] = 'Small'
    combined.loc[combined['Outlet_Identifier'] == 'OUT045', 'Outlet_Size'] = 'Small'
    combined.loc[combined['Outlet_Identifier'] == 'OUT017', 'Outlet_Size'] = 'Small'
    
    # Fill any remaining missing Outlet_Size with mode
    combined['Outlet_Size'].fillna(combined['Outlet_Size'].mode()[0], inplace=True)
 
    # Standardize Item_Fat_Content values
    fat_content_map = {
        'LF': 'Low Fat',
        'low fat': 'Low Fat',
        'reg': 'Regular'
    }
    combined['Item_Fat_Content'] = combined['Item_Fat_Content'].replace(fat_content_map)
    
    # Create Item_Category from first two characters of Item_Identifier
    combined['Item_Category'] = combined['Item_Identifier'].str.slice(0, 2)
    combined['Item_Category'] = combined['Item_Category'].map({
        'FD': 'Food',
        'NC': 'Non-Consumable',
        'DR': 'Drinks'
    })
    
    # Set 'Not Applicable' for Non-Consumable items
    combined.loc[combined['Item_Category'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Not Applicable'
    
    # Fix Item_Visibility=0 with mean visibility of same Item_Type
    item_visibility = {}
    for item_type in combined['Item_Type'].unique():
        type_mask = (combined['Item_Type'] == item_type) & (combined['Item_Visibility'] > 0)
        visibility = combined.loc[type_mask, 'Item_Visibility']
        if len(visibility) > 0:
            item_visibility[item_type] = visibility.mean()
        else:
            item_visibility[item_type] = combined['Item_Visibility'].mean()
    
    # Replace 0 visibility values
    zero_mask = combined['Item_Visibility'] == 0
    for idx, row in combined[zero_mask].iterrows():
        item_type = row['Item_Type']
        combined.at[idx, 'Item_Visibility'] = item_visibility[item_type]
    
    # Create log transformation of Item_Visibility
    combined['Item_Visibility_Log'] = np.log1p(combined['Item_Visibility'])
    
    # Create MRP bins
    combined['Item_MRP_Bins'] = pd.qcut(combined['Item_MRP'], 4, labels=['Budget', 'Regular', 'Premium', 'Luxury'])

    # Create Outlet_Years feature and drop Outlet_Establishment_Year
    combined['Outlet_Years'] = 2013 - combined['Outlet_Establishment_Year']
    
    # Create derived features
    combined['Price_Per_Weight'] = combined['Item_MRP'] / combined['Item_Weight']
    combined['Visibility_to_MRP'] = combined['Item_Visibility'] / combined['Item_MRP']
    
    # Update Item_Identifier to be category names
    # First save the original for other operations
    combined['Original_Item_Identifier'] = combined['Item_Identifier']
    # Then map to category names
    combined['Item_Identifier'] = combined['Item_Identifier'].str.slice(0, 2)
    combined['Item_Identifier'] = combined['Item_Identifier'].map({
        'FD': 'Food',
        'NC': 'Non_Consumable',
        'DR': 'Drinks'
    })
    
    # One-hot encode categorical variables
    categorical_cols = [
        'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
        'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type',
        'Item_Category', 'Item_MRP_Bins', 'Item_Identifier'
    ]
    
    combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)
    
    # Drop unnecessary columns
    combined.drop(['Original_Item_Identifier', 'Outlet_Establishment_Year'], axis=1, inplace=True)
    
    # Split back into train and test
    train_processed = combined.iloc[:len(train)]
    test_processed = combined.iloc[len(train):]
    
    # Drop target column from test
    test_processed = test_processed.drop('Item_Outlet_Sales', axis=1)
    
    # Ensure test has all columns from train
    missing_cols = set(train_processed.columns) - set(test_processed.columns) - {'Item_Outlet_Sales'}
    for col in missing_cols:
        test_processed[col] = 0
    
    # Ensure column order matches
    test_processed = test_processed[train_processed.drop('Item_Outlet_Sales', axis=1).columns]
    
    return train_processed, test_processed, original_test

def train_models_and_predict(train_processed, test_processed):
    X_train = train_processed.drop('Item_Outlet_Sales', axis=1)
    y_train = train_processed['Item_Outlet_Sales']
    X_test = test_processed
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        objective='reg:squarederror',
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_preds = xgb_model.predict(X_test_scaled)
    
    # Train Gradient Boosting model
    print("Training Gradient Boosting model...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    gb_preds = gb_model.predict(X_test_scaled)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    
    # Evaluate with cross-validation
    print("\nModel Evaluation:")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'XGBoost': xgb_model,
        'Gradient Boosting': gb_model,
        'Random Forest': rf_model
    }
    
    # RMSE scores for each model
    rmse_scores = {}
    for name, model in models.items():
        scores = -cross_val_score(
            model, X_train_scaled, y_train,
            scoring='neg_root_mean_squared_error',
            cv=cv, n_jobs=-1
        )
        rmse_scores[name] = scores.mean()
        print(f"{name} CV RMSE: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Calculate weights based on inverse RMSE
    inverse_rmse = {name: 1/score for name, score in rmse_scores.items()}
    total = sum(inverse_rmse.values())
    weights = {name: score/total for name, score in inverse_rmse.items()}
    
    print("\nEnsemble Weights:")
    for name, weight in weights.items():
        print(f"{name}: {weight:.4f}")
    
    # Create weighted ensemble prediction
    ensemble_preds = (
        weights['XGBoost'] * xgb_preds +
        weights['Gradient Boosting'] * gb_preds +
        weights['Random Forest'] * rf_preds
    )
    
    ensemble_preds = np.maximum(ensemble_preds, 0)
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    top_features = feature_importance.head(15)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 15 Features by Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    return ensemble_preds, feature_importance


def main():
    print("Loading data...")
    train = pd.read_csv("train_v9rqX0R.csv")
    test = pd.read_csv("test_AbJTz2l.csv")
    
    print(f"Training data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")
    
    print("\nPreprocessing data...")
    train_processed, test_processed, original_test = preprocess_data(train, test)

    print("\nTraining models...")
    ensemble_preds, feature_importance = train_models_and_predict(train_processed, test_processed)
    

    submission = pd.DataFrame({
        'Item_Identifier': original_test['Item_Identifier'],
        'Outlet_Identifier': original_test['Outlet_Identifier'],
        'Item_Outlet_Sales': ensemble_preds
    })
    
    submission.to_csv('improved_ensemble_submission.csv', index=False)
    print("\nSubmission file saved as 'improved_ensemble_submission.csv'")
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
if __name__ == "__main__":
    main()