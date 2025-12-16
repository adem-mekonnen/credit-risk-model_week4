import pytest
import pandas as pd
import sys
import os

# Add src to path so we can import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import create_aggregate_features

def test_aggregate_features():
    # Mock Data
    data = {
        'AccountId': [1, 1, 2],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-01'],
        'Amount': [100, 200, 50],
        'Value': [100, 200, 50]
    }
    df = pd.DataFrame(data)
    
    # Run Function
    result = create_aggregate_features(df)
    
    # Assertions
    assert result.shape[0] == 2 # Should correspond to 2 Accounts
    assert 'TotalAmount' in result.columns
    assert result.loc[result['AccountId'] == 1, 'TotalAmount'].values[0] == 300