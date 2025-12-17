import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# NEW IMPORT: Use category_encoders for stable WoE
from category_encoders import WOEEncoder 
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ... (Keep load_data, create_aggregate_features, create_rfm_target functions unchanged) ...

def load_data(filepath):
    """Loads data with error handling."""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def create_aggregate_features(df):
    """Aggregates transaction data to AccountId level (Task 3)."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    aggs = {'Amount': ['sum', 'mean', 'std', 'count'], 'Value': ['sum', 'mean']}
    df_agg = df.groupby('AccountId').agg(aggs).reset_index()
    df_agg.columns = ['AccountId', 'TotalAmount', 'AvgAmount', 'StdAmount', 'TxCount', 'TotalValue', 'AvgValue']
    df_agg = df_agg.fillna(0)
    return df_agg

def create_rfm_target(df):
    """Creates Proxy Target Variable using RFM Clustering (Task 4)."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('AccountId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={'TransactionStartTime': 'Recency', 'TransactionId': 'Frequency', 'Amount': 'Monetary'})
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    cluster_summary = rfm.groupby('Cluster').mean()
    high_risk_cluster = cluster_summary.sort_values(by=['Recency', 'Frequency'], ascending=[False, True]).index[0]
    
    rfm['is_high_risk'] = rfm['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)
    return rfm[['is_high_risk']]


def run_pipeline(input_path, output_path):
    
    try:
        df = load_data(input_path)
    except FileNotFoundError:
        logging.error("Pipeline cannot run because the raw data file was not found.")
        return

    df_agg = create_aggregate_features(df)
    target_df = create_rfm_target(df)
    
    woe_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    cat_features = df.groupby('AccountId')[woe_cols].agg(
        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
    ).reset_index()

    final_df = df_agg.merge(cat_features, on='AccountId').merge(target_df, on='AccountId')
    
    # ------------------------------------------------------------------
    # FIX: Use WOEEncoder from category_encoders (stable)
    # ------------------------------------------------------------------
    # The WOEEncoder requires the categorical column names to be passed
    # It converts them to float (log-odds), which solves the numeric value error.
    clf_woe = WOEEncoder(cols=woe_cols, handle_missing='value', handle_unknown='value')
    y = final_df['is_high_risk']
    
    # Fit and Transform
    df_woe = clf_woe.fit_transform(final_df, y)
    
    # WoEEncoder adds new columns and modifies the dataframe in place/as a copy.
    # We rename the new WoE columns to clean up.
    woe_mapped_cols = {col: f'{col}_WOE' for col in woe_cols}
    df_woe = df_woe.rename(columns=woe_mapped_cols)
    
    # Select only the original numeric/aggregate features, the new WOE features, and the target
    final_cols = list(df_agg.columns) + list(woe_mapped_cols.values()) + ['is_high_risk']
    final_df = df_woe[final_cols]
    
    # Remove the redundant Value columns as per EDA insight
    final_df = final_df.drop(columns=['TotalValue', 'AvgValue'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logging.info(f"WOE features created using category_encoders. Processed data saved to {output_path}")

if __name__ == "__main__":
    run_pipeline("data/raw/data.csv", "data/processed/train_data.csv")