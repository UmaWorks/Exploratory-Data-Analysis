import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import warnings
warnings.filterwarnings("ignore")

class AdvancedPreprocessor(BaseEstimator, TransformerMixin):
    """Advanced preprocessing with feature engineering"""
    
    def __init__(self, item_weight_strategy='item_identifier', outlet_size_strategy='outlet_type'):
        self.item_weight_strategy = item_weight_strategy
        self.outlet_size_strategy = outlet_size_strategy
        self.item_weights = {}
        self.type_weights = {}
        self.outlet_sizes = {}
        self.item_visibility = {}
        self.outlet_avg_sales = {}
        self.mrp_stats = {}
        
    def fit(self, X, y=None):
        """Learn statistics from training data"""
        if self.item_weight_strategy == 'item_identifier':
            for item_id in X['Item_Identifier'].unique():
                item_mask = X['Item_Identifier'] == item_id
                weights = X.loc[item_mask, 'Item_Weight'].dropna()
                if len(weights) > 0:
                    self.item_weights[item_id] = weights.mean()
        
        # Store item weights per type
        for item_type in X['Item_Type'].unique():
            type_mask = X['Item_Type'] == item_type
            weights = X.loc[type_mask, 'Item_Weight'].dropna()
            if len(weights) > 0:
                self.type_weights[item_type] = weights.mean()
        
        # Store outlet sizes per type
        if self.outlet_size_strategy == 'outlet_type':
            for outlet_type in X['Outlet_Type'].unique():
                outlet_mask = X['Outlet_Type'] == outlet_type
                sizes = X.loc[outlet_mask, 'Outlet_Size'].dropna()
                if len(sizes) > 0 and not sizes.empty:
                    self.outlet_sizes[outlet_type] = sizes.mode()[0]
        
        # Store mean visibility per item type
        for item_type in X['Item_Type'].unique():
            type_mask = (X['Item_Type'] == item_type) & (X['Item_Visibility'] > 0)
            visibility = X.loc[type_mask, 'Item_Visibility']
            if len(visibility) > 0:
                self.item_visibility[item_type] = visibility.mean()
        
        # Store mean visibility for fallback
        self.global_visibility = X.loc[X['Item_Visibility'] > 0, 'Item_Visibility'].mean()
        
        # Store MRP statistics for normalization and binning
        self.mrp_stats['min'] = X['Item_MRP'].min()
        self.mrp_stats['max'] = X['Item_MRP'].max()
        self.mrp_stats['mean'] = X['Item_MRP'].mean()
        self.mrp_stats['std'] = X['Item_MRP'].std()
        
        # Store outlet average sales if target is available
        if 'Item_Outlet_Sales' in X.columns and any(X['Item_Outlet_Sales'] > 0):
            for outlet in X['Outlet_Identifier'].unique():
                outlet_mask = X['Outlet_Identifier'] == outlet
                if outlet_mask.sum() > 0 and any(X.loc[outlet_mask, 'Item_Outlet_Sales'] > 0):
                    self.outlet_avg_sales[outlet] = X.loc[outlet_mask, 'Item_Outlet_Sales'].mean()
        
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
      
        # 1. Fill missing Item_Weight values
        missing_weight = X_transformed['Item_Weight'].isna()
        
        for idx in X_transformed[missing_weight].index:
            item_id = X_transformed.loc[idx, 'Item_Identifier']
            item_type = X_transformed.loc[idx, 'Item_Type']
            
            # Try item identifier based imputation
            if item_id in self.item_weights:
                X_transformed.at[idx, 'Item_Weight'] = self.item_weights[item_id]
            # Fallback to item type based imputation
            elif item_type in self.type_weights:
                X_transformed.at[idx, 'Item_Weight'] = self.type_weights[item_type]
            # Final fallback to global mean
            else:
                X_transformed.at[idx, 'Item_Weight'] = np.mean(list(self.item_weights.values()))
        
        # 2. Fill missing Outlet_Size values based on specific Outlet_Identifier
        # Direct assignment for specific outlets as requested
        X_transformed.loc[X_transformed['Outlet_Identifier'] == 'OUT010', 'Outlet_Size'] = 'Small'
        X_transformed.loc[X_transformed['Outlet_Identifier'] == 'OUT045', 'Outlet_Size'] = 'Small'
        X_transformed.loc[X_transformed['Outlet_Identifier'] == 'OUT017', 'Outlet_Size'] = 'Small'
        
        # Fill remaining missing Outlet_Size values
        missing_size = X_transformed['Outlet_Size'].isna()
        
        for idx in X_transformed[missing_size].index:
            outlet_type = X_transformed.loc[idx, 'Outlet_Type']
            
            # Try outlet type based imputation
            if outlet_type in self.outlet_sizes:
                X_transformed.at[idx, 'Outlet_Size'] = self.outlet_sizes[outlet_type]
            # Fallback to most common size
            else:
                X_transformed.at[idx, 'Outlet_Size'] = max(self.outlet_sizes.values(), key=list(self.outlet_sizes.values()).count)
                
        # ---- FEATURE CLEANING ----
        
        # 1. Standardize Item_Fat_Content
        fat_content_map = {
            'LF': 'Low Fat',
            'low fat': 'Low Fat',
            'reg': 'Regular',
            'regular': 'Regular'
        }
        X_transformed['Item_Fat_Content'] = X_transformed['Item_Fat_Content'].replace(fat_content_map)
        
        # 2. Convert Item_Identifier to first 2 characters and map to category names
        X_transformed['Original_Item_Identifier'] = X_transformed['Item_Identifier'].copy()
        X_transformed['Item_Identifier'] = X_transformed['Item_Identifier'].apply(lambda x: x[0:2])
        category_map = {
            'FD': 'Food',
            'NC': 'Non_Consumable',
            'DR': 'Drinks'
        }
        X_transformed['Item_Identifier'] = X_transformed['Item_Identifier'].map(category_map)
        
        # Create Item_Category with the same mapping for backwards compatibility
        X_transformed['Item_Category'] = X_transformed['Original_Item_Identifier'].str.slice(0, 2)
        X_transformed['Item_Category'] = X_transformed['Item_Category'].map(category_map)
        
        # Set logical fat content for Non-Consumables
        X_transformed.loc[X_transformed['Item_Category'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Not Applicable'
        
        # ---- HANDLE ZERO VISIBILITY VALUES ----
        zero_visibility = X_transformed['Item_Visibility'] == 0
        
        for idx in X_transformed[zero_visibility].index:
            item_type = X_transformed.loc[idx, 'Item_Type']
            
            # Use item type based visibility
            if item_type in self.item_visibility:
                X_transformed.at[idx, 'Item_Visibility'] = self.item_visibility[item_type]
            # Fallback to global visibility
            else:
                X_transformed.at[idx, 'Item_Visibility'] = self.global_visibility
        
        # ---- ADVANCED FEATURE ENGINEERING ----
        
        # 1. Transform visibility to handle skewness
        X_transformed['Item_Visibility_Log'] = np.log1p(X_transformed['Item_Visibility'])
        
        # 2. Create MRP features
        # Normalized MRP
        X_transformed['Item_MRP_Normalized'] = (X_transformed['Item_MRP'] - self.mrp_stats['mean']) / self.mrp_stats['std']
        
        # MRP bins with equal frequency
        X_transformed['Item_MRP_Bins'] = pd.qcut(
            X_transformed['Item_MRP'], 
            q=4, 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # 3. Create years of operation feature
        X_transformed['Outlet_Years'] = 2013 - X_transformed['Outlet_Establishment_Year']
        
        # 4. Create price per weight ratio
        X_transformed['Price_Per_Weight'] = X_transformed['Item_MRP'] / X_transformed['Item_Weight']
        
        # 5. Create visibility to MRP ratio
        X_transformed['Visibility_to_MRP'] = X_transformed['Item_Visibility'] / X_transformed['Item_MRP']
        
        # 6. Create outlet-specific features
        # Outlet average sales (Target encoding)
        if self.outlet_avg_sales:
            X_transformed['Outlet_Avg_Sales'] = X_transformed['Outlet_Identifier'].map(self.outlet_avg_sales)
            # Fill missing with global average if column exists
            if 'Outlet_Avg_Sales' in X_transformed.columns:
                global_avg = np.mean(list(self.outlet_avg_sales.values())) if self.outlet_avg_sales else 0
                X_transformed['Outlet_Avg_Sales'].fillna(global_avg, inplace=True)
        
        # 7. Create interaction features - simple version to avoid dimensionality explosion
        X_transformed['Outlet_Type_X_Location'] = X_transformed['Outlet_Type'] + '_' + X_transformed['Outlet_Location_Type']
        X_transformed['Item_Category_X_Outlet_Type'] = X_transformed['Item_Category'] + '_' + X_transformed['Outlet_Type']
        
        # ---- ENCODING CATEGORICAL VARIABLES ----
        # Prepare for one-hot encoding
        categorical_cols = [
            'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type',
            'Item_Category', 'Item_MRP_Bins', 'Outlet_Type_X_Location',
            'Item_Category_X_Outlet_Type', 'Item_Identifier'
        ]
        
        # Apply one-hot encoding
        X_encoded = pd.get_dummies(X_transformed, columns=categorical_cols, drop_first=False)
        
        # ---- DROP UNNECESSARY COLUMNS ----
        cols_to_drop = [
            'Original_Item_Identifier', 'Outlet_Establishment_Year'  # Drop Establishment Year as requested
        ]
        X_final = X_encoded.drop(cols_to_drop, axis=1)
        
        return X_final
    
def create_lightgbm_model():
    """Create optimized LightGBM model that avoids no-split warnings"""
    return lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,        # Reduced from previous 
        max_depth=5,          # Reduced to avoid overfitting
        min_child_samples=5,  # Smaller sample needed per leaf
        subsample=0.7,        # Sample 70% of data per tree
        colsample_bytree=0.7, # Sample 70% of features per tree
        reg_alpha=0.03,       # Increased L1 regularization
        reg_lambda=0.3,       # Increased L2 regularization
        min_split_gain=0.01,  # Minimum gain needed for split
        min_data_in_leaf=10,  # Minimum data points in leaf
        random_state=42,
        verbose=-1            # Suppress warnings
    )


class WeightedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Weighted ensemble of multiple models with optimized weights"""
    
    def __init__(self, models, cv=5):
        self.models = models
        self.cv = cv
        self.trained_models = []
        self.weights = None
    
    def fit(self, X, y):
        """Fit all models and calculate optimal weights"""
        self.trained_models = []
        
        print("Training individual models...")
        # Train each model on the full dataset
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model_clone = clone(model)
            model_clone.fit(X, y)
            self.trained_models.append((name, model_clone))
        
        # Calculate weights based on cross-validation performance
        print("Calculating optimal weights based on cross-validation...")
        cv_errors = {}
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train on this fold
                m = clone(model)
                m.fit(X_train, y_train)
                
                # Score on validation fold
                preds = m.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                scores.append(rmse)
            
            # Store average RMSE for this model
            cv_errors[name] = np.mean(scores)
            print(f"  {name} cross-validation RMSE: {cv_errors[name]:.4f}")
        
        inverse_squared_errors = {name: 1/(error**2) for name, error in cv_errors.items()}
        sum_weights = sum(inverse_squared_errors.values())
        self.weights = {name: w/sum_weights for name, w in inverse_squared_errors.items()}
        
        print("Final model weights:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")
        
        return self
    
    def predict(self, X):
        """Generate weighted ensemble prediction"""
        # Collect predictions from all models
        all_preds = {}
        for name, model in self.trained_models:
            all_preds[name] = model.predict(X)
        

        ensemble_pred = np.zeros(X.shape[0])
        for name, preds in all_preds.items():
            ensemble_pred += self.weights[name] * preds
        
        return np.maximum(ensemble_pred, 0)


def train_and_predict(train_path, test_path):
    """Complete pipeline from data loading to final prediction"""
    # Load data
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    original_test = test.copy()
    
    print(f"Training data: {train.shape}")
    print(f"Test data: {test.shape}")
    
    # Add dummy target to test for consistent processing
    test['Item_Outlet_Sales'] = 0
    
    # Advanced preprocessing
    print("\nApplying advanced preprocessing...")
    preprocessor = AdvancedPreprocessor()
    preprocessor.fit(train)
    
    train_processed = preprocessor.transform(train)
    test_processed = preprocessor.transform(test)
    
    # Prepare training and test data
    X_train = train_processed.drop('Item_Outlet_Sales', axis=1)
    y_train = train_processed['Item_Outlet_Sales']
    X_test = test_processed.drop('Item_Outlet_Sales', axis=1)
    
    # Ensure column consistency
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Apply feature scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Define models
    print("\nDefining models...")
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1,
            objective='reg:squarederror',
            random_state=42
        ),
        'LightGBM': create_lightgbm_model(),
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
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
    }
    
    # Train weighted ensemble
    print("\nTraining weighted ensemble model...")
    ensemble = WeightedEnsembleRegressor(models=models, cv=5)
    ensemble.fit(X_train_scaled, y_train)
    
    # Generate predictions
    print("\nGenerating predictions...")
    ensemble_predictions = ensemble.predict(X_test_scaled)
    
    # Create submission file
    submission = pd.DataFrame({
        'Item_Identifier': original_test['Item_Identifier'],
        'Outlet_Identifier': original_test['Outlet_Identifier'],
        'Item_Outlet_Sales': ensemble_predictions
    })
    

    submission.to_csv('optimized_ensemble_submission.csv', index=False)
    print("\nSubmission file saved as 'optimized_ensemble_submission.csv'")
    
    print("\nGenerating individual model predictions for comparison...")
    for name, model in ensemble.trained_models:
        preds = np.maximum(model.predict(X_test_scaled), 0)  # Ensure non-negative
        model_submission = pd.DataFrame({
            'Item_Identifier': original_test['Item_Identifier'],
            'Outlet_Identifier': original_test['Outlet_Identifier'],
            'Item_Outlet_Sales': preds
        })
        filename = f"{name.lower()}_submission.csv"
        model_submission.to_csv(filename, index=False)
        print(f"  Saved {name} predictions to {filename}")
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    # Get feature importance from XGBoost model
    xgb_model = [model for name, model in ensemble.trained_models if name == 'XGBoost'][0]
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Features by Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    print("\nFeature importance plot saved to 'feature_importance.png'")
    print(f"\nTop 10 most important features:\n{feature_importance.head(10)}")
    
    return submission, feature_importance


if __name__ == "__main__":
    submission, feature_importance = train_and_predict(
        "train_v9rqX0R.csv", 
        "test_AbJTz2l.csv"
    )
