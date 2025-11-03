# experiments/evaluate_cicids.py
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from evaluation.cicids_tester import CICIDS2017Tester
from evaluation.benchmark_comparator import BenchmarkComparator
from piperag_integration.parallel_pipeline import PipeRAGParallelPipeline
from streamrag.pathway_vector_db import PathwayStreamRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./outputs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main evaluation pipeline
    1. Load CICIDS-2017 data
    2. Test baseline LLM
    3. Test PipeRAG
    4. Compare results
    """
    
    logger.info("Starting CICIDS-2017 Evaluation")
    logger.info("="*70)
    
    # Initialize components
    tester = CICIDS2017Tester()
    comparator = BenchmarkComparator()
    
    # ==================== LOAD DATA ====================
    logger.info("Step 1: Loading CICIDS-2017 dataset")
    
    # Use one of your data files
    data_file = "./data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        logger.info("Available files in ./data/:")
        for f in os.listdir("./data"):
            logger.info(f"  - {f}")
        return
    
    cicids_df = tester.load_cicids_data(
        csv_path=data_file,
        sample_size=200  # Start with 200 for testing
    )
    
    if cicids_df is None or len(cicids_df) == 0:
        logger.error("Failed to load data")
        return
    
    # ==================== TEST BASELINE ====================
    logger.info("\nStep 2: Testing Baseline LLM (GPT-2)")
    
    baseline_results = tester.test_baseline_llm(
        flows_df=cicids_df,
        model_name="gpt2"
    )
    baseline_metrics = tester.calculate_metrics(baseline_results, "Baseline LLM")
    
    # ==================== TEST PIPERAG ====================
    logger.info("\nStep 3: Testing PipeRAG")
    
    try:
        # Initialize vector DB
        sessions_parquet = "./outputs/preprocessed/sessions.parquet"
        if not os.path.exists(sessions_parquet):
            logger.warning(f"Sessions parquet not found at {sessions_parquet}")
            logger.warning("Skipping PipeRAG test. Run preprocessing first.")
            piperag_results = None
            piperag_metrics = None
        else:
            vector_db = PathwayStreamRAG(sessions_parquet)
            piperag_model = PipeRAGParallelPipeline(
                vector_db=vector_db,
                model_name="gpt2"
            )
            
            piperag_results = tester.test_piperag(
                flows_df=cicids_df,
                piperag_model=piperag_model
            )
            piperag_metrics = tester.calculate_metrics(piperag_results, "PipeRAG")
    
    except Exception as e:
        logger.error(f"Error testing PipeRAG: {e}")
        piperag_results = None
        piperag_metrics = None
    
    # ==================== COMPARE ====================
    logger.info("\nStep 4: Comparing Results")
    
    if piperag_results is not None and piperag_metrics is not None:
        tester.compare_models(baseline_results, piperag_results)
        
        # Save comparison
        comparison = comparator.save_results(
            baseline_df=baseline_results,
            piperag_df=piperag_results,
            baseline_metrics=baseline_metrics,
            piperag_metrics=piperag_metrics,
            dataset_name="CICIDS2017"
        )
        
        report = comparator.generate_report(comparison)
        logger.info(report)
        
        # Save report
        with open('./outputs/results/evaluation_report.txt', 'w') as f:
            f.write(report)
    else:
        logger.warning("Could not compare - PipeRAG test failed")
        baseline_results.to_csv('./outputs/results/baseline_cicids2017_results.csv', index=False)
    
    logger.info("\n✅ Evaluation Complete!")
    logger.info(f"Results saved to: ./outputs/results/")
    logger.info(f"Log saved to: ./outputs/evaluation.log")


if __name__ == "__main__":
    main()
