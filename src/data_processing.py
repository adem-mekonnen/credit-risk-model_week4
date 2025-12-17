import pytest
import pandas as pd
import numpy as np
import sys
import os

# Adjust path for CI to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import create_aggregate_features

def test_aggregate_features_sum_count():
    # Mock Data
    data = {
        'AccountId': [1, 1, 2],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-01'],
        'Amount': [100, 200, 50],
        'Value': [100, 200, 50]
    }
    df = pd.DataFrame(data)
    
    result = create_aggregate_features(df)
    
    # Test 1: Shape check
    assert result.shape[0] == 2 
    # Test 2: Sum Amount check (Customer 1 = 100+200=300)
    assert result.loc[result['AccountId'] == 1, 'TotalAmount'].values[0] == 300
    # Test 3: Transaction Count check (Customer 2 = 1)
    assert result.loc[result['AccountId'] == 2, 'TxCount'].values[0] == 1

def test_aggregate_features_std_imputation():
    # Mock data where one customer has only one transaction (should result in NaN, then 0 imputation)
    data = {
        'AccountId': [10, 20],
        'TransactionStartTime': ['2023-01-01', '2023-01-02'],
        'Amount': [500, 100],
        'Value': [500, 100]
    }
    df = pd.DataFrame(data)
    
    result = create_aggregate_features(df)

    # Test 4: StdDev for customer 20 (single transaction) must be 0 after imputation
    assert result.loc[result['AccountId'] == 20, 'StdAmount'].values[0] == 0
    # Test 5: All required columns are present
    expected_cols = ['AccountId', 'TotalAmount', 'StdAmount', 'TxCount']
    assert all(col in result.columns for col in expected_cols)