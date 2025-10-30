"""StreamRAG with Pathway integration"""
import pathway as pw
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
import numpy as np
import pandas as pd
import time
from typing import List, Tuple

class PathwayStreamRAG:
    def __init__(self, sessions_parquet, embedding_dim=401):
        self.sessions_parquet = sessions_parquet
        self.embedding_dim = embedding_dim
        
        # Load preprocessed sessions
        self.sessions_df = pd.read_parquet(sessions_parquet)
        print(f"[PathwayStreamRAG] Loaded {len(self.sessions_df)} sessions")
        
        # Storage for streaming
        self.embeddings = []
        self.texts = []
        self.metadata = []
        self.current_idx = 0
    
    def start_streaming(self, rate_limit=10.0):
        """Simulate streaming ingestion at rate_limit sessions/sec"""
        print(f"[PathwayStreamRAG] Starting streaming at {rate_limit} sessions/sec...")
        
        interval = 1.0 / rate_limit
        for idx, row in self.sessions_df.iterrows():
            # Add to vector DB
            embedding = np.array(row['embedding'])
            text = row['narrative']
            meta = {
                'session_id': row['session_id'],
                'start_time': row['session_start_time'],
                'label': row['label']
            }
            
            self.embeddings.append(embedding)
            self.texts.append(text)
            self.metadata.append(meta)
            self.current_idx += 1
            
            time.sleep(interval)
            
            if self.current_idx % 100 == 0:
                print(f"[Streaming] Ingested {self.current_idx}/{len(self.sessions_df)} sessions")
    
    def retrieve(self, query_embedding, top_k=5) -> List[Tuple]:
        """Retrieve top-k similar sessions"""
        if not self.embeddings:
            return []
        
        embeddings_array = np.array(self.embeddings)
        similarities = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                similarities[idx],
                self.texts[idx],
                self.metadata[idx]
            ))
        
        return results
    
    def get_current_size(self):
        return len(self.embeddings)
