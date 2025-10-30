# preprocessing/session_builder.py
"""Session building from process_traffic_sessions.ipynb"""
import pandas as pd
from typing import List, Dict, Any

class SessionBuilder:
    def __init__(self, timeout_seconds=900):
        self.timeout_seconds = timeout_seconds
    
    def sessionize(self, df):
        """Group flows into sessions - simplified for incomplete CSV"""
        print(f"[SessionBuilder] Creating sessions (simplified mode)...")
        print(f"[SessionBuilder] Total rows: {len(df)}")
        
        # Since Source IP/Port columns are missing, group every 10 rows as a session
        sessions = []
        chunk_size = 10
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            session = self._aggregate_session(chunk, f"session_{i//chunk_size}")
            sessions.append(session)
            
            if len(sessions) % 1000 == 0:
                print(f"[SessionBuilder] Processed {len(sessions)} sessions...")
        
        sessions_df = pd.DataFrame(sessions)
        print(f"[SessionBuilder] Created {len(sessions_df)} sessions from {len(df)} flows")
        return sessions_df
    
    def _aggregate_session(self, flows_df, session_id):
        """Aggregate flows into single session"""
        # Find label column
        label_col = None
        for variant in ['Label', ' Label', 'Label ', ' Label ']:
            if variant in flows_df.columns:
                label_col = variant
                break
        
        # Get label
        if label_col:
            label_mode = flows_df[label_col].mode()
            label = label_mode[0] if len(label_mode) > 0 else 'BENIGN'
        else:
            label = 'BENIGN'
        
        # Get total packets
        total_fwd = 0
        total_bwd = 0
        for variant in [' Total Fwd Packets', 'Total Fwd Packets']:
            if variant in flows_df.columns:
                total_fwd = flows_df[variant].sum()
                break
        
        for variant in [' Total Backward Packets', 'Total Backward Packets']:
            if variant in flows_df.columns:
                total_bwd = flows_df[variant].sum()
                break
        
        session = {
            'session_id': session_id,
            'session_start_time': flows_df['parsed_timestamp'].min(),
            'session_end_time': flows_df['parsed_timestamp'].max(),
            'flow_count': len(flows_df),
            'total_packets': int(total_fwd + total_bwd),
            'total_bytes': 0,
            'source_ip': 'unknown',
            'dest_ip': 'unknown',
            'protocol': 0,
            'label': label,
            'narrative': f"Network flow session with {len(flows_df)} entries, Label: {label}"
        }
        
        return session
