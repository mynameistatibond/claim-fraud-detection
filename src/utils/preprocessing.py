"""
Preprocessing utilities for insurance claims dataset.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, OrdinalEncoder, OneHotEncoder
import numpy as np


def get_column_groups(X_train):
    """
    Define column groups for preprocessing.
    
    Args:
        X_train: Training DataFrame
    
    Returns:
        dict with keys: num_cols, ohe_cols, ord_cols, ord_categories
    
    Example:
        >>> groups = get_column_groups(X_train)
        >>> print(f"Numeric: {len(groups['num_cols'])}")
        Numeric: 31
    """
    # Auto-detect column types
    num_auto = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_auto = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Binary flags should be numeric (0/1), not one-hot encoded
    bool_like = [
        'collision_type_missing', 'property_damage_missing',
        'police_report_available_missing', 'authorities_contacted_missing',
        'incident_weekend', 'is_holiday', 'holiday_window_2d'
    ]
    
    # Remove from categorical, add to numeric
    for c in bool_like:
        if c in cat_auto:
            cat_auto.remove(c)
        if c in X_train.columns and c not in num_auto:
            num_auto.append(c)
            
    # EXCLUDE LEAKY FEATURES
    # 'insured_hobbies' is identified as a leaky feature that skews results
    # UPDATE: User requested to KEEP insured_hobbies for tree models
    drop_cols = [] # 'insured_hobbies' was here
    for c in drop_cols:
        if c in cat_auto:
            cat_auto.remove(c)
        if c in num_auto:
            num_auto.remove(c)
    
    # Ordered categoricals (have natural order)
    ord_cols = ['vehicle_age_bucket', 'vehicle_tier']
    ord_categories = [
        ['new', 'mid', 'old'],      # vehicle_age_bucket: newer → older
        ['low', 'mid', 'premium']   # vehicle_tier: low → high
    ]
    
    # Filter to only include columns that exist in X_train
    ord_cols = [c for c in ord_cols if c in X_train.columns]
    ohe_cols = [c for c in cat_auto if c not in ord_cols and c in X_train.columns]
    num_cols = [c for c in num_auto if c in X_train.columns]
    
    return {
        'num_cols': num_cols,
        'ohe_cols': ohe_cols,
        'ord_cols': ord_cols,
        'ord_categories': ord_categories
    }


def get_preprocessor(X_train=None, strategy='robust', column_groups=None):
    """
    Returns configured ColumnTransformer for preprocessing.
    
    Args:
        X_train: Training data (needed if column_groups not provided)
        strategy: 'robust' (RobustScaler) or 'standard' (StandardScaler)
        column_groups: Dict from get_column_groups() (optional, auto-computed if None)
    
    Returns:
        ColumnTransformer configured for the dataset
    
    Example:
        >>> preprocessor = get_preprocessor(X_train, strategy='robust')
        >>> preprocessor.fit(X_train)
    
    Note:
        - RobustScaler recommended for data with outliers (uses median, IQR)
        - StandardScaler for normally distributed data (uses mean, std)
    """
    if column_groups is None:
        assert X_train is not None, "Must provide either X_train or column_groups"
        column_groups = get_column_groups(X_train)
    
    # Choose scaler based on strategy
    scaler = RobustScaler() if strategy == 'robust' else StandardScaler()
    
    transformers = []
    
    # Add numeric scaler if we have numeric columns
    if column_groups['num_cols']:
        transformers.append(("num", scaler, column_groups['num_cols']))
    
    # Add ordinal encoder if we have ordered categorical columns
    if column_groups['ord_cols']:
        transformers.append((
            "ord",
            OrdinalEncoder(
                categories=column_groups['ord_categories'],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ),
            column_groups['ord_cols']
        ))
    
    # Add one-hot encoder for remaining categoricals
    if column_groups['ohe_cols']:
        transformers.append((
            "ohe",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            column_groups['ohe_cols']
        ))
    
    return ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )
