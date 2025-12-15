import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    """Aggregates transaction data to Customer (AccountId) level."""
    logging.info("Creating aggregate features...")
    try:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        aggs = {
            'Amount': ['sum', 'mean', 'std', 'count'],
            'Value': ['sum', 'mean']
        }
        
        # Group by AccountId
        df_agg = df.groupby('AccountId').agg(aggs).reset_index()
        df_agg.columns = ['AccountId', 'TotalAmount', 'AvgAmount', 'StdAmount', 'TxCount', 'TotalValue', 'AvgValue']
        
        # Fill NaN (std dev is NaN if only 1 transaction)
        df_agg = df_agg.fillna(0)
        return df_agg
    except Exception as e:
        logging.error(f"Error in feature aggregation: {e}")
        raise

def create_rfm_target(df):
    """Creates Proxy Target Variable using RFM Clustering."""
    logging.info("Engineering Proxy Target (RFM)...")
    try:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        
        # RFM Calculation
        rfm = df.groupby('AccountId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': 'sum'
        }).rename(columns={'TransactionStartTime': 'Recency', 'TransactionId': 'Frequency', 'Amount': 'Monetary'})
        
        # Scale Data
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)
        
        # K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Assign Risk Label (High Recency + Low Frequency = High Risk)
        cluster_summary = rfm.groupby('Cluster').mean()
        high_risk_cluster = cluster_summary.sort_values(by=['Recency'], ascending=False).index[0]
        
        rfm['is_high_risk'] = rfm['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)
        
        return rfm[['is_high_risk']]
    except Exception as e:
        logging.error(f"Error in RFM creation: {e}")
        raise

def run_pipeline(input_path, output_path):
    try:
        df = load_data(input_path)
        
        # Feature Engineering
        df_agg = create_aggregate_features(df)
        
        # Categorical Features (Mode)
        cat_features = df.groupby('AccountId')[['ProductCategory', 'ChannelId', 'PricingStrategy']].agg(
            lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
        ).reset_index()
        
        # Target Generation
        target_df = create_rfm_target(df)
        
        # Merge
        final_df = df_agg.merge(cat_features, on='AccountId').merge(target_df, on='AccountId')
        
        # Encoding
        le = LabelEncoder()
        for col in ['ProductCategory', 'ChannelId', 'PricingStrategy']:
            final_df[col] = le.fit_transform(final_df[col].astype(str))
            
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline("data/raw/data.csv", "data/processed/train_data.csv")