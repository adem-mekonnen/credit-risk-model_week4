import pytest
import pandas as pd
import numpy as np
# We use try/except import to handle path differences in CI vs Local
try:
    from src.data_processing import create_aggregate_features
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.data_processing import create_aggregate_features

def test_aggregate_logic():
    # Mock data resembling Xente structure
    data = {
        'AccountId': [101, 101, 102],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Amount': [500, 500, 100],
        'Value': [500, 500, 100]
    }
    df = pd.DataFrame(data)
    
    # Run the function
    result = create_aggregate_features(df)
    
    # Assertions
    assert result.shape[0] == 2  # Should be 2 unique customers (101 and 102)
    assert 'TotalAmount' in result.columns
    assert result.loc[result['AccountId'] == 101, 'TotalAmount'].values[0] == 1000