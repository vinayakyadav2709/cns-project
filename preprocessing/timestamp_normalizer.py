"""Timestamp normalization from normalization.ipynb"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TimestampNormalizer:
    def __init__(self, random_delay_max=5.0):
        self.random_delay_max = random_delay_max
    
    def normalize(self, df):
        """Normalize timestamps and add ingestion delays"""
        print(f"[Normalizer] Processing {len(df)} rows...")
        
        # Parse timestamp column
        timestamp_col = self._find_timestamp_column(df)
        
        if timestamp_col:
            print(f"[Normalizer] Found timestamp column: '{timestamp_col}'")
            df['parsed_timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
        else:
            # CICIDS2017 sometimes lacks timestamps - generate synthetic ones
            print("[Normalizer] No timestamp column found, generating synthetic timestamps...")
            base_time = datetime(2017, 7, 7, 14, 0, 0)  # Friday afternoon
            df['parsed_timestamp'] = [base_time + timedelta(seconds=i*0.1) for i in range(len(df))]
        
        # Remove invalid timestamps
        df = df.dropna(subset=['parsed_timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('parsed_timestamp').reset_index(drop=True)
        
        # Add simulated ingestion delay (0-5 seconds)
        df['ingestion_delay_sec'] = np.random.uniform(0, self.random_delay_max, len(df))
        df['ingestion_timestamp'] = df['parsed_timestamp'] + pd.to_timedelta(df['ingestion_delay_sec'], unit='s')
        
        # Extract temporal features
        df['hour'] = df['parsed_timestamp'].dt.hour
        df['day_of_week'] = df['parsed_timestamp'].dt.dayofweek
        df['minute'] = df['parsed_timestamp'].dt.minute
        
        print(f"[Normalizer] Normalized {len(df)} rows successfully")
        return df
    
    def _find_timestamp_column(self, df):
        """Find timestamp column dynamically - handle spaces and variations"""
        # Print first 10 columns for debugging
        print(f"[Normalizer] First 10 columns: {df.columns.tolist()[:10]}")
        
        # Check for timestamp-like columns (with/without spaces)
        candidates = [
            'Timestamp', 'timestamp', 'time', 'Time', 'datetime',
            ' Timestamp', ' timestamp', ' Time', ' time',  # Leading space
            'Timestamp ', 'timestamp ', 'Time ', 'time '   # Trailing space
        ]
        
        for col in candidates:
            if col in df.columns:
                return col
        
        # Try finding any column containing 'time' (case insensitive)
        for col in df.columns:
            if 'time' in col.lower():
                print(f"[Normalizer] Found time-like column: '{col}'")
                return col
        
        # No timestamp found
        return None
