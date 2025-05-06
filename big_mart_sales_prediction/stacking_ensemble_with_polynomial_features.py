#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


class RobustTargetTransformer(BaseEstimator, TransformerMixin):
    """Advanced target transformer with outlier handling"""
    
    def __init__(self, method='quantile', n_quantiles=1000, output_distribution='normal'):
        self.method = method
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.transformer = QuantileTransformer(
            n_quantiles=n_quantiles, 
            output_distribution=output_distribution
        )
        
    def fit(self, y):
        # Convert to numpy array if it's pandas Series
        if isinstance(y, pd.Series):
            y = y.values
            
        # Cap outliers at the 99th percentile
        cap_value = np.percentile(y, 99)
        y_capped = np.minimum(y, cap_value)
        self.transformer.fit(y_capped.reshape(-1, 1))
        return self
        
    def transform(self, y):
        # Convert to numpy array if it's pandas Series
        if isinstance(y, pd.Series):
            y = y.values
            
        cap_value = np.percentile(y, 99) if len(y) > 100 else np.max(y)
        y_capped = np.minimum(y, cap_value)
        return self.transformer.transform(y_capped.reshape(-1, 1)).flatten()
        
    def inverse_transform(self, y):
        # Convert to numpy array if it's pandas Series
        if isinstance(y, pd.Series):
            y = y.values
            
        return self.transformer.inverse_transform(y.reshape(-1, 1)).flatten()



def preprocess_data(train, test, target_transform=True):
    """Advanced preprocessing function with feature engineering"""
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
    
    # Fill missing Outlet_Size
    outlet_sizes = combined.groupby('Outlet_Type')['Outlet_Size'].apply(
        lambda x: x.mode()[0] if not x.mode().empty else "Medium"
    )
    
    for idx in combined[combined['Outlet_Size'].isna()].index:
        outlet_type = combined.loc[idx, 'Outlet_Type']
        if outlet_type in outlet_sizes:
            combined.at[idx, 'Outlet_Size'] = outlet_sizes[outlet_type]
        else:
            combined.at[idx, 'Outlet_Size'] = "Medium"  # Default
    
   
    fat_map = {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}
    combined['Item_Fat_Content'] = combined['Item_Fat_Content'].replace(fat_map)
    
    # Extract Item Category from identifier
    combined['Item_Category'] = combined['Item_Identifier'].str.slice(0, 2)
    combined['Item_Category'] = combined['Item_Category'].map({
        'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'
    })
    
    # Set logical fat content for Non-Consumables
    combined.loc[combined['Item_Category'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Not Applicable'
    

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
    
    
    # 1. Visibility transformations
    combined['Item_Visibility_Log'] = np.log1p(combined['Item_Visibility'])
    combined['Item_Visibility_Sqrt'] = np.sqrt(combined['Item_Visibility'])
    combined['Item_Visibility_Cubert'] = np.cbrt(combined['Item_Visibility'])
    
    # 2. MRP features
    combined['Item_MRP_Sqrt'] = np.sqrt(combined['Item_MRP'])
    combined['Item_MRP_Squared'] = combined['Item_MRP'] ** 2
    
    # MRP bins with equal frequency
    combined['Item_MRP_Bins'] = pd.qcut(
        combined['Item_MRP'], 
        q=5, 
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # 3. Outlet features
    combined['Outlet_Years'] = 2013 - combined['Outlet_Establishment_Year']
    combined['Outlet_Years_Squared'] = combined['Outlet_Years'] ** 2
    
    # Create Era categorical feature
    combined['Establishment_Era'] = pd.cut(
        combined['Outlet_Establishment_Year'],
        bins=[1980, 1990, 2000, 2010, 2013],
        labels=['Very Old', 'Old', 'Recent', 'New']
    )
    
    # 4. Price and weight features
    combined['Price_Per_Weight'] = combined['Item_MRP'] / combined['Item_Weight']
    combined['Price_Per_Weight_Log'] = np.log1p(combined['Price_Per_Weight'])
    combined['Weight_To_MRP_Ratio'] = combined['Item_Weight'] / combined['Item_MRP']
    
    # 5. Visibility ratio features
    combined['Visibility_To_MRP'] = combined['Item_Visibility'] / combined['Item_MRP']
    combined['Visibility_To_Weight'] = combined['Item_Visibility'] / combined['Item_Weight']
    
    # 6. Target encoding features
    # Only calculate from training data to avoid leakage
    train_mask = combined.index < len(train)
    
    # Outlet average sales
    outlet_avg_sales = combined[train_mask].groupby('Outlet_Identifier')['Item_Outlet_Sales'].mean()
    combined['Outlet_Avg_Sales'] = combined['Outlet_Identifier'].map(outlet_avg_sales)
    combined['Outlet_Avg_Sales'].fillna(combined[train_mask]['Item_Outlet_Sales'].mean(), inplace=True)
    
    # Item Type average sales
    type_avg_sales = combined[train_mask].groupby('Item_Type')['Item_Outlet_Sales'].mean()
    combined['Item_Type_Avg_Sales'] = combined['Item_Type'].map(type_avg_sales)
    combined['Item_Type_Avg_Sales'].fillna(combined[train_mask]['Item_Outlet_Sales'].mean(), inplace=True)
    
    # Category average sales
    cat_avg_sales = combined[train_mask].groupby('Item_Category')['Item_Outlet_Sales'].mean()
    combined['Item_Category_Avg_Sales'] = combined['Item_Category'].map(cat_avg_sales)
    combined['Item_Category_Avg_Sales'].fillna(combined[train_mask]['Item_Outlet_Sales'].mean(), inplace=True)
    
    # 7. Outlet identifier features
    # Count encoding for outlets
    outlet_counts = combined['Outlet_Identifier'].value_counts().to_dict()
    combined['Outlet_Count'] = combined['Outlet_Identifier'].map(outlet_counts)
    
    # 8. Item category and type features
    # Create Item Type & Fat Content combination
    combined['Item_Category_X_Outlet_Type'] = combined['Item_Category'] + '_' + combined['Outlet_Type']
    combined['Item_Category_X_Fat'] = combined['Item_Category'] + '_' + combined['Item_Fat_Content']
    combined['Outlet_Type_X_Location'] = combined['Outlet_Type'] + '_' + combined['Outlet_Location_Type']
    combined['Outlet_Size_X_Location'] = combined['Outlet_Size'] + '_' + combined['Outlet_Location_Type']
    
    # 9. Tree-specific interaction features
    combined['MRP_X_Weight'] = combined['Item_MRP'] * combined['Item_Weight']
    combined['MRP_X_Visibility'] = combined['Item_MRP'] * combined['Item_Visibility']
    combined['MRP_X_Outlet_Years'] = combined['Item_MRP'] * combined['Outlet_Years']
    
    
    # Prepare for one-hot encoding
    cat_cols = [
        'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
        'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type',
        'Item_Category', 'Item_MRP_Bins', 'Establishment_Era',
        'Item_Category_X_Outlet_Type', 'Item_Category_X_Fat',
        'Outlet_Type_X_Location', 'Outlet_Size_X_Location'
    ]
    
    # Apply one-hot encoding
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=False)
    
    # ---- DROP UNNECESSARY COLUMNS ----
    cols_to_drop = ['Item_Identifier', 'Outlet_Establishment_Year']
    combined.drop(cols_to_drop, axis=1, inplace=True)
    
    # Split back into train and test
    train_processed = combined.iloc[:len(train)]
    test_processed = combined.iloc[len(train):]
    
    # Remove target column from test
    X_test = test_processed.drop('Item_Outlet_Sales', axis=1)
    y_train = train_processed['Item_Outlet_Sales']
    X_train = train_processed.drop('Item_Outlet_Sales', axis=1)
    
    # Apply target transformation if requested
    target_transformer = None
    if target_transform:
        target_transformer = RobustTargetTransformer(method='quantile')
        y_train = target_transformer.fit_transform(y_train)
    
    print(f"Preprocessing complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, original_test, target_transformer



def advanced_preprocess(train, test):
    """Enhanced preprocessing with polynomial features"""
    # Start with the base preprocessing
    X_train, y_train, X_test, original_test, _ = preprocess_data(train, test, target_transform=False)
    
    # Apply robust target transformation
    robust_transformer = RobustTargetTransformer(method='quantile')
    y_train_transformed = robust_transformer.fit_transform(y_train)
    
    # Add polynomial features for key variables
    print("Adding polynomial features...")
    
    # Identify key numerical features
    num_features = ['Item_MRP', 'Item_Visibility', 'Item_Weight', 'Outlet_Years', 
                   'Price_Per_Weight', 'Visibility_To_MRP', 'Item_Visibility_Log',
                   'Outlet_Avg_Sales', 'Item_Type_Avg_Sales']
    
    # Create polynomial terms only for these important features
    for i, feat1 in enumerate(num_features):
        if feat1 in X_train.columns:
            # Square terms
            X_train[f'{feat1}_squared'] = X_train[feat1] ** 2
            X_test[f'{feat1}_squared'] = X_test[feat1] ** 2
            
            # Interaction terms
            for feat2 in num_features[i+1:]:
                if feat2 in X_train.columns:
                    X_train[f'{feat1}_x_{feat2}'] = X_train[feat1] * X_train[feat2]
                    X_test[f'{feat1}_x_{feat2}'] = X_test[feat1] * X_test[feat2]
    
    print(f"After polynomial features: Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, y_train_transformed, X_test, original_test, robust_transformer


def create_stacking_ensemble():
    """Create advanced stacking ensemble with optimized base models"""
    # Define base models
    base_models = [
        ('xgb', xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.005,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.05,
            reg_alpha=0.05,
            reg_lambda=1,
            random_state=42
        )),
        ('lgb', lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=6,
            boosting_type='dart',  # Using dart for better regularization
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            verbose=-1
        )),
        ('gbm', GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=5,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        ))
    ]
    
    # Define meta model (Ridge regression works well as a meta-learner)
    meta_model = Ridge(alpha=1.0)
    
    # Create stacking regressor with 5-fold CV
    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    return stacking_regressor


def train_advanced_model(X_train, y_train, X_test, target_transformer):
    """Train stacking ensemble and generate predictions"""
    # Create stacking ensemble
    ensemble = create_stacking_ensemble()
    
    # Train ensemble
    print("Training stacking ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    print("Generating predictions...")
    preds = ensemble.predict(X_test)
    
    # Inverse transform predictions
    print("Applying inverse transformation...")
    final_preds = target_transformer.inverse_transform(preds)
    
    # Ensure non-negative values
    final_preds = np.maximum(final_preds, 0)
    
    return final_preds, ensemble


def analyze_feature_importance(X_train, ensemble):
    """Analyze feature importance from ensemble models"""
    try:
        # Get feature importance from XGBoost base model
        xgb_model = ensemble.estimators_[0][1]  # XGBoost is the first model
        
        if hasattr(xgb_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': xgb_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 10))
            top_features = feature_importance.head(20)
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title('Top 20 Features by Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            
            print("\nTop 10 most important features:")
            print(feature_importance.head(10))
            
            return feature_importance
        else:
            print("Model doesn't support feature importance extraction")
            return None
    except Exception as e:
        print(f"Error extracting feature importance: {e}")
        return None


def advanced_stacking_pipeline(train_path, test_path):
    """Complete advanced pipeline with stacking ensemble"""
    # Load data
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"Original training data: {train.shape}")
    print(f"Original test data: {test.shape}")
    
    # Enhanced preprocessing with polynomial features
    X_train, y_train, X_test, original_test, target_transformer = advanced_preprocess(train, test)
    
    # Train model and generate predictions
    predictions, ensemble = train_advanced_model(X_train, y_train, X_test, target_transformer)
    
    # Create submission file
    submission = pd.DataFrame({
        'Item_Identifier': original_test['Item_Identifier'],
        'Outlet_Identifier': original_test['Outlet_Identifier'],
        'Item_Outlet_Sales': predictions
    })
    
    # Save submission
    submission.to_csv('advanced_stacking_submission.csv', index=False)
    print("\nSubmission file saved as 'advanced_stacking_submission.csv'")
    
    # Analyze feature importance
    analyze_feature_importance(X_train, ensemble)
    
    return submission, ensemble


if __name__ == "__main__":
    submission, ensemble = advanced_stacking_pipeline(
        "train_v9rqX0R.csv", 
        "test_AbJTz2l.csv"
    )
