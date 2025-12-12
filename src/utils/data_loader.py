"""
Data loading utilities for insurance claims dataset.
"""
from pathlib import Path
import pandas as pd


def load_insurance_data(dataset_type='preprocessed', verbose=True):
    """
    Load insurance claims dataset.
    
    Args:
        dataset_type: 'preprocessed' or 'engineered'
        verbose: Print loading info
    
    Returns:
        df: DataFrame with insurance claims data
    
    Example:
        >>> df = load_insurance_data('preprocessed')
        ✓ Loading preprocessed dataset: insurance_claims_preprocessed_no_hobbies.csv
          Shape: (1000, 51)
    """
    # Determine root directory (handles both notebook and script contexts)
    ROOT = Path.cwd().parent if 'notebooks' in str(Path.cwd()) else Path.cwd()
    PROC = ROOT / "data" / "processed"
    
    if dataset_type == 'preprocessed':
        DATA = PROC / "insurance_claims_preprocessed_no_hobbies.csv"
    elif dataset_type == 'engineered':
        DATA = PROC / "insurance_claims_engineered_final.csv"
    elif dataset_type == 'trees':
        DATA = PROC / "preprocesed_for_trees.csv"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'preprocessed', 'engineered', or 'trees'.")
    
    assert DATA.exists(), f"File not found: {DATA}"
    
    if verbose:
        print(f"✓ Loading {dataset_type} dataset: {DATA.name}")
    
    df = pd.read_csv(DATA)
    
    if verbose:
        print(f"  Shape: {df.shape}")
        if 'target' in df.columns:
            target_dist = df['target'].value_counts().to_dict()
            fraud_rate = df['target'].mean()
            print(f"  Target distribution: {target_dist}")
            print(f"  Fraud rate: {fraud_rate:.1%}")
    
    return df
