import pytest
import pandas as pd
import sys
import os

# 1. Adjust path for CI to import src modules (KEEP THIS)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Import the module itself, NOT the function, to break the loop
# This should resolve the circular import error
import src.data_processing as dp 


def test_aggregate_features_sum_count():
    # Mock Data
    data = {
        'AccountId': [1, 1, 2],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-01'],
        'Amount': [100, 200, 50],
        'Value': [100, 200, 50]
    }
    df = pd.DataFrame(data)
    
    # CALL THE FUNCTION using the module prefix (dp.)
    result = dp.create_aggregate_features(df) 
    
    # Assertions
    assert result.shape[0] == 2 
    assert 'TotalAmount' in result.columns
    assert result.loc[result['AccountId'] == 1, 'TotalAmount'].values[0] == 300
    assert result.loc[result['AccountId'] == 2, 'TxCount'].values[0] == 1

def test_aggregate_features_std_imputation():
    # Mock data where one customer has only one transaction
    data = {
        'AccountId': [10, 20],
        'TransactionStartTime': ['2023-01-01', '2023-01-02'],
        'Amount': [500, 100],
        'Value': [500, 100]
    }
    df = pd.DataFrame(data)
    
    # CALL THE FUNCTION using the module prefix (dp.)
    result = dp.create_aggregate_features(df)

    # StdDev for customer 20 (single transaction) must be 0 after imputation
    assert result.loc[result['AccountId'] == 20, 'StdAmount'].values[0] == 0
    # Check column existence
    expected_cols = ['AccountId', 'TotalAmount', 'StdAmount', 'TxCount']
    assert all(col in result.columns for col in expected_cols)