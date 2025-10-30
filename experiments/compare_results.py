"""Compare PipeRAG vs Baseline"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load results
    baseline = pd.read_csv("../outputs/results/baseline_results.csv")
    piperag = pd.read_csv("../outputs/results/piperag_results.csv")
    
    # Compute metrics[SessionBuilder] Processed 10000 sessions...
[SessionBuilder] Processed 11000 sessions...
[SessionBuilder] Processed 12000 sessions...
[SessionBuilder] Processed 13000 sessions...
[SessionBuilder] Processed 14000 sessions...
[SessionBuilder] Processed 15000 sessions...
[SessionBuilder] Processed 16000 sessions...
[SessionBuilder] Processed 17000 sessions...
[SessionBuilder] Processed 18000 sessions...
[SessionBuilder] Processed 19000 sessions...
[SessionBuilder] Processed 20000 sessions...
[SessionBuilder] Processed 21000 sessions...
[SessionBuilder] Processed 22000 sessions...
[SessionBuilder] Created 22575 sessions from 225745 flows
[TemporalAligner] Creating 404-dim embeddings...
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 706/706 [00:35<00:00, 19.67it/s]
[TemporalAligner] Created embeddings with shape (22575, 401)
[Pipeline] Saved 22575 sessions to outputs/preprocessed/sessions.parquet
(.venv) falcon@fedora:~/Projects/Microsoft/cybersec-piperag$ python experiments/run_baseline.py
Traceback (most recent call last):
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/experiments/run_baseline.py", line 7, in <module>
    from models.baseline_lightning import BaselineLongContextModel
ModuleNotFoundError: No module named 'models'
(.venv) falcon@fedora:~/Projects/Microsoft/cybersec-piperag$ python experiments/run_baseline.py
Traceback (most recent call last):
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/experiments/run_baseline.py", line 7, in <module>
    from models.baseline_lightning import BaselineLongContextModel
ModuleNotFoundError: No module named 'models'
(.venv) falcon@fedora:~/Projects/Microsoft/cybersec-piperag$ python experiments/run_baseline.py
/home/falcon/Projects/Microsoft/.venv/lib/python3.10/site-packages/lightning_fabric/__init__.py:40: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
Traceback (most recent call last):
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/experiments/run_baseline.py", line 10, in <module>
    from models.baseline_lightning import BaselineLongContextModel
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/models/__init__.py", line 2, in <module>
    from .piperag_lightning import PipeRAGLightningModule
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/models/piperag_lightning.py", line 3, in <module>
    from piperag_integration.parallel_pipeline import PipeRAGParallelPipeline
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/piperag_integration/__init__.py", line 2, in <module>
    from .parallel_pipeline import PipeRAGParallelPipeline
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/piperag_integration/parallel_pipeline.py", line 6, in <module>
    from performance_model import FixedIntervalModel
ModuleNotFoundError: No module named 'performance_model'
(.venv) falcon@fedora:~/Projects/Microsoft/cybersec-piperag$ python experiments/run_baseline.py
/home/falcon/Projects/Microsoft/.venv/lib/python3.10/site-packages/lightning_fabric/__init__.py:40: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
Traceback (most recent call last):
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/experiments/run_baseline.py", line 10, in <module>
    from models.baseline_lightning import BaselineLongContextModel
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/models/__init__.py", line 2, in <module>
    from .piperag_lightning import PipeRAGLightningModule
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/models/piperag_lightning.py", line 3, in <module>
    from piperag_integration.parallel_pipeline import PipeRAGParallelPipeline
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/piperag_integration/__init__.py", line 2, in <module>
    from .parallel_pipeline import PipeRAGParallelPipeline
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/piperag_integration/parallel_pipeline.py", line 6, in <module>
    from performance_model import FixedIntervalModel
ModuleNotFoundError: No module named 'performance_model'
(.venv) falcon@fedora:~/Projects/Microsoft/cybersec-piperag$ cd /home/falcon/Projects/Microsoft/cybersec-piperag
PYTHONPATH=. python experiments/run_baseline.py
/home/falcon/Projects/Microsoft/.venv/lib/python3.10/site-packages/lightning_fabric/__init__.py:40: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
Traceback (most recent call last):
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/experiments/run_baseline.py", line 10, in <module>
    from models.baseline_lightning import BaselineLongContextModel
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/models/__init__.py", line 2, in <module>
    from .piperag_lightning import PipeRAGLightningModule
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/models/piperag_lightning.py", line 3, in <module>
    from piperag_integration.parallel_pipeline import PipeRAGParallelPipeline
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/piperag_integration/__init__.py", line 2, in <module>
    from .parallel_pipeline import PipeRAGParallelPipeline
  File "/home/falcon/Projects/Microsoft/cybersec-piperag/piperag_integration/parallel_pipeline.py", line 6, in <module>
    from performance_model import FixedIntervalModel
ModuleNotFoundError: No module named 'performance_model'
(.venv) falcon@fedora:~/Projects/Microsoft/cybersec-piperag$ 
    baseline_metrics = {
        'correct': baseline['correct'].sum(),
        'total': len(baseline),
        'avg_latency': baseline['latency'].mean()
    }
    
    piperag_metrics = {
        'correct': piperag['correct'].sum(),
        'total': len(piperag),
        'avg_latency': piperag['latency'].mean()
    }
    
    baseline_acc = baseline_metrics['correct'] / baseline_metrics['total']
    piperag_acc = piperag_metrics['correct'] / piperag_metrics['total']
    
    print("="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Baseline Accuracy: {baseline_acc:.3f}")
    print(f"PipeRAG Accuracy: {piperag_acc:.3f}")
    print(f"Baseline Avg Latency: {baseline_metrics['avg_latency']:.3f}s")
    print(f"PipeRAG Avg Latency: {piperag_metrics['avg_latency']:.3f}s")
    print(f"Speedup: {baseline_metrics['avg_latency'] / piperag_metrics['avg_latency']:.2f}x")
    
    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    models = ['Baseline\n(Long Context)', 'PipeRAG\n(Streaming)']
    accuracies = [baseline_acc, piperag_acc]
    ax1.bar(models, accuracies, color=['#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Detection Accuracy')
    ax1.set_ylim([0, 1])
    
    # Latency comparison
    latencies = [baseline_metrics['avg_latency'], piperag_metrics['avg_latency']]
    ax2.bar(models, latencies, color=['#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Latency (seconds)')
    ax2.set_title('Average Response Time')
    
    plt.tight_layout()
    plt.savefig('../outputs/charts/comparison.png', dpi=300)
    print("\nChart saved to outputs/charts/comparison.png")

if __name__ == "__main__":
    main()
