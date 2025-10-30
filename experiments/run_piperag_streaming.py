"""Run PipeRAG streaming experiment"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import time
from streamrag.pathway_vector_db import PathwayStreamRAG
from models.piperag_lightning import PipeRAGLightningModule
from evaluation.anomaly_detector import AnomalyDetector
import threading

def main():
    print("="*60)
    print("PIPERAG STREAMING EXPERIMENT")
    print("="*60)
    
    # 1. Load preprocessed sessions
    sessions_parquet = "../outputs/preprocessed/sessions.parquet"
    sessions_df = pd.read_parquet(sessions_parquet)
    print(f"Loaded {len(sessions_df)} sessions")
    
    # 2. Initialize StreamRAG
    vector_db = PathwayStreamRAG(sessions_parquet)
    
    # 3. Start streaming in background
    stream_rate = 10.0  # sessions/sec
    streaming_thread = threading.Thread(
        target=vector_db.start_streaming,
        args=(stream_rate,),
        daemon=True
    )
    streaming_thread.start()
    
    # Wait for some data to load
    time.sleep(5)
    print(f"Vector DB size: {vector_db.get_current_size()}")
    
    # 4. Initialize PipeRAG model
    model = PipeRAGLightningModule(vector_db)
    
    # 5. Run detection queries
    detector = AnomalyDetector()
    results = []
    
    # Sample 100 test queries
    test_sessions = sessions_df.sample(n=min(100, len(sessions_df)), random_state=42)
    
    print("\nRunning PipeRAG detection...")
    for idx, row in test_sessions.iterrows():
        query = f"Is this network activity malicious? {row['narrative']}"
        
        start_time = time.time()
        response = model(query)
        latency = time.time() - start_time
        
        result = detector.detect_from_response(response, row['label'])
        result['latency'] = latency
        results.append(result)
        
        if len(results) % 10 == 0:
            print(f"Processed {len(results)}/{len(test_sessions)}")
    
    # 6. Compute metrics
    metrics = detector.compute_metrics(results)
    avg_latency = sum(r['latency'] for r in results) / len(results)
    
    print("\n" + "="*60)
    print("PIPERAG RESULTS")
    print("="*60)
    print(f"Total Anomalies Detected: {metrics['total_detected']}")
    print(f"True Attacks (Ground Truth): {metrics['total_actual']}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print(f"Average Latency: {avg_latency:.3f}s")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("../outputs/results/piperag_results.csv", index=False)
    print("\nResults saved to outputs/results/piperag_results.csv")

if __name__ == "__main__":
    main()
