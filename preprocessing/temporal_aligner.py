"""Temporal alignment from TemporalAlignment.ipynb"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

class TemporalAligner:
    def __init__(self, embedding_dim=384):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = embedding_dim
        self.scaler = StandardScaler()
    
    def align(self, sessions_df):
        """Create hybrid temporal-semantic embeddings"""
        print(f"[TemporalAligner] Creating {self.embedding_dim + 20}-dim embeddings...")
        
        # 1. Semantic embeddings from narratives
        narratives = sessions_df['narrative'].tolist()
        semantic_embeddings = self.embedder.encode(narratives, show_progress_bar=True)
        
        # 2. Temporal features (sinusoidal encoding)
        sessions_df['hour_sin'] = np.sin(2 * np.pi * sessions_df['session_start_time'].dt.hour / 24)
        sessions_df['hour_cos'] = np.cos(2 * np.pi * sessions_df['session_start_time'].dt.hour / 24)
        sessions_df['day_sin'] = np.sin(2 * np.pi * sessions_df['session_start_time'].dt.dayofweek / 7)
        sessions_df['day_cos'] = np.cos(2 * np.pi * sessions_df['session_start_time'].dt.dayofweek / 7)
        
        temporal_features = sessions_df[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']].values
        
        # 3. IP feature hashing (simple hash to 10-dim)
        ip_features = self._hash_ip_features(sessions_df)
        
        # 4. Statistical features
        stat_features = sessions_df[['flow_count', 'total_packets', 'total_bytes']].values
        stat_features = self.scaler.fit_transform(stat_features)
        
        # Combine all features
        hybrid_embeddings = np.concatenate([
            semantic_embeddings,  # 384-dim
            temporal_features,    # 4-dim
            ip_features,          # 10-dim
            stat_features         # 3-dim
        ], axis=1)
        
        sessions_df['embedding'] = list(hybrid_embeddings)
        print(f"[TemporalAligner] Created embeddings with shape {hybrid_embeddings.shape}")
        
        return sessions_df
    
    def _hash_ip_features(self, sessions_df, dim=10):
        """Simple IP address feature hashing"""
        ip_hashes = []
        for _, row in sessions_df.iterrows():
            src_hash = hash(str(row.get('source_ip', ''))) % dim
            dst_hash = hash(str(row.get('dest_ip', ''))) % dim
            
            feature = np.zeros(dim)
            feature[src_hash] = 1
            feature[dst_hash] = 1
            ip_hashes.append(feature)
        
        return np.array(ip_hashes)
