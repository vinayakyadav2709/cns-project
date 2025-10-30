"""Orchestrates all preprocessing steps"""
import pandas as pd
import argparse
from pathlib import Path
from timestamp_normalizer import TimestampNormalizer
from session_builder import SessionBuilder
from temporal_aligner import TemporalAligner

class PreprocessPipeline:
    def __init__(self):
        self.normalizer = TimestampNormalizer()
        self.sessionizer = SessionBuilder(timeout_seconds=900)
        self.aligner = TemporalAligner()
    
    def run(self, input_csv, output_parquet):
        print(f"[Pipeline] Starting preprocessing: {input_csv}")
        
        # Load raw data
        df = pd.read_csv(input_csv)
        print(f"[Pipeline] Loaded {len(df)} rows")
        
        # Step 1: Normalize timestamps
        df = self.normalizer.normalize(df)
        
        # Step 2: Create sessions
        sessions_df = self.sessionizer.sessionize(df)
        
        # Step 3: Temporal alignment
        sessions_df = self.aligner.align(sessions_df)
        
        # Save to parquet
        output_path = Path(output_parquet)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sessions_df.to_parquet(output_path, index=False)
        
        print(f"[Pipeline] Saved {len(sessions_df)} sessions to {output_parquet}")
        return sessions_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output Parquet file')
    args = parser.parse_args()
    
    pipeline = PreprocessPipeline()
    pipeline.run(args.input, args.output)
