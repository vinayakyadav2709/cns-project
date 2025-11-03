# evaluation/benchmark_comparator.py
import pandas as pd
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BenchmarkComparator:
    """Generate comparison reports and save results"""
    
    def __init__(self, results_dir: str = "./outputs/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, baseline_df: pd.DataFrame, piperag_df: pd.DataFrame,
                    baseline_metrics: dict, piperag_metrics: dict,
                    dataset_name: str = "CICIDS2017"):
        """Save all results to CSV and JSON"""
        
        # Save dataframes
        baseline_df.to_csv(
            self.results_dir / f"baseline_{dataset_name.lower()}_results.csv",
            index=False
        )
        piperag_df.to_csv(
            self.results_dir / f"piperag_{dataset_name.lower()}_results.csv",
            index=False
        )
        
        # Save metrics
        comparison = {
            'dataset': dataset_name,
            'baseline': baseline_metrics,
            'piperag': piperag_metrics,
            'improvement': {
                'accuracy': piperag_metrics['accuracy'] - baseline_metrics['accuracy'],
                'precision': piperag_metrics['precision'] - baseline_metrics['precision'],
                'f1_score': piperag_metrics['f1_score'] - baseline_metrics['f1_score'],
                'latency_speedup': baseline_metrics['avg_latency'] / piperag_metrics['avg_latency']
            }
        }
        
        with open(self.results_dir / f"comparison_{dataset_name.lower()}.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Results saved to {self.results_dir}")
        
        return comparison
    
    def generate_report(self, comparison: dict) -> str:
        """Generate human-readable report"""
        report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    CICIDS-2017 EVALUATION REPORT                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📊 BASELINE LLM (No RAG)
├─ Accuracy:          {comparison['baseline']['accuracy']:.4f}
├─ Precision:         {comparison['baseline']['precision']:.4f}
├─ Recall:            {comparison['baseline']['recall']:.4f}
├─ F1-Score:          {comparison['baseline']['f1_score']:.4f}
├─ Avg Latency:       {comparison['baseline']['avg_latency']:.3f}s
└─ Throughput:        {comparison['baseline']['throughput']:.2f} flows/sec

🎯 PIPERAG (With RAG)
├─ Accuracy:          {comparison['piperag']['accuracy']:.4f}
├─ Precision:         {comparison['piperag']['precision']:.4f}
├─ Recall:            {comparison['piperag']['recall']:.4f}
├─ F1-Score:          {comparison['piperag']['f1_score']:.4f}
├─ Avg Latency:       {comparison['piperag']['avg_latency']:.3f}s
└─ Throughput:        {comparison['piperag']['throughput']:.2f} flows/sec

✨ IMPROVEMENTS
├─ Accuracy Gain:     {comparison['improvement']['accuracy']:+.4f} ({comparison['improvement']['accuracy']*100:+.2f}%)
├─ Precision Gain:    {comparison['improvement']['precision']:+.4f}
├─ F1-Score Gain:     {comparison['improvement']['f1_score']:+.4f}
└─ Speedup:           {comparison['improvement']['latency_speedup']:.2f}x faster

{"✅ PipeRAG is better!" if comparison['improvement']['accuracy'] > 0 else "⚠️  Baseline is better"}
"""
        return report
