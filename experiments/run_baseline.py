"""Run baseline long context experiment"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import time
from models.baseline_lightning import BaselineLongContextModel
from evaluation.anomaly_detector import AnomalyDetector

# Rest of the code stays the same...

def main():
    print("="*60)
    print("BASELINE LONG CONTEXT EXPERIMENT")
    print("="*60)
    
    # 1. Load preprocessed sessions
    sessions_parquet = "outputs/preprocessed/sessions.parquet"
    sessions_df = pd.read_parquet(sessions_parquet)
    print(f"Loaded {len(sessions_df)} sessions")
    
    # 2. Initialize baseline model
    model = BaselineLongContextModel()
    model.eval()
    
    # 3. Run detection queries
    detector = AnomalyDetector()
    results = []
    
    # Sample 100 test queries
    test_sessions = sessions_df.sample(n=min(100, len(sessions_df)), random_state=42)
    
    # Create full context (all logs as single text)
    full_logs_text = "\n".join(sessions_df['narrative'].tolist())
    print(f"Full context length: {len(full_logs_text)} chars")
    
    print("\nRunning baseline detection...")
    for idx, row in test_sessions.iterrows():
        query = f"Is this network activity malicious? {row['narrative']}"
        
        start_time = time.time()
        response = model.generate_full_context(full_logs_text, query, max_length=8192)
        latency = time.time() - start_time
        
        result = detector.detect_from_response(response, row['label'])
        result['latency'] = latency
        results.append(result)
        
        if len(results) % 10 == 0:
            print(f"Processed {len(results)}/{len(test_sessions)}")
    
    # 4. Compute metrics
    metrics = detector.compute_metrics(results)
    avg_latency = sum(r['latency'] for r in results) / len(results)
    
    print("\n" + "="*60)
    print("BASELINE RESULTS")
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
    results_df.to_csv("../outputs/results/baseline_results.csv", index=False)
    print("\nResults saved to outputs/results/baseline_results.csv")

if __name__ == "__main__":
    main()
