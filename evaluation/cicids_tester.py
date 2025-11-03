# evaluation/cicids_tester.py - COMPLETE WORKING VERSION

import pandas as pd
import numpy as np
from typing import Dict
import time
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CICIDS2017Tester:
    """Test PipeRAG vs Baseline LLM on CICIDS-2017"""
    
    def __init__(self):
        pass
        
    # evaluation/cicids_tester.py - FINAL FIX

    def load_cicids_data(self, csv_path: str, sample_size: int = 1000) -> pd.DataFrame:
        """Load CICIDS-2017 CSV - DON'T USE include_groups=False"""
        logger.info(f"Loading CICIDS-2017 data from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None
        
        logger.info(f"✓ CSV loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        df.columns = df.columns.str.strip()
        logger.info(f"✓ Column names stripped")
        
        label_col = df.columns[-1]
        logger.info(f"✓ Label column identified: '{label_col}'")
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        logger.info(f"✓ Replaced inf/NaN values with 0")
        
        attack_counts = df[label_col].value_counts()
        logger.info(f"✓ Attack distribution found:")
        for attack_type, count in attack_counts.items():
            logger.info(f"  - {attack_type}: {count} flows")
        
        # SAMPLE - Remove include_groups parameter completely
        if len(df) > sample_size:
            logger.info(f"✓ Sampling {sample_size} flows (from {len(df)})")
            
            # Stratified sampling that PRESERVES the label column
            sampled_dfs = []
            for label_value, group in df.groupby(label_col):
                n_samples = min(len(group), max(1, sample_size // 2))
                sampled_dfs.append(group.sample(n=n_samples, random_state=42))
            
            df = pd.concat(sampled_dfs, ignore_index=True)
        
        logger.info(f"✓ Final dataset: {len(df)} flows\n")
        
        # Rename
        df.rename(columns={label_col: 'Label'}, inplace=True)
        
        # Verify
        if 'Label' not in df.columns:
            raise ValueError(f"Label column missing! Columns: {list(df.columns)}")
        
        return df

    
    def flow_to_narrative(self, flow: pd.Series) -> str:
        """Convert flow to narrative"""
        try:
            duration = flow.get('Flow Duration', 0)
            fwd = int(flow.get('Total Fwd Packets', 0))
            bwd = int(flow.get('Total Backward Packets', 0))
            rate = flow.get('Flow Bytes/s', 0)
            syn = int(flow.get('SYN Flag Count', 0))
            rst = int(flow.get('RST Flag Count', 0))
            ack = int(flow.get('ACK Flag Count', 0))
            return f"Flow: {duration:.1f}s | Packets={fwd+bwd} | Rate={rate:.0f}B/s | Flags(SYN={syn} RST={rst} ACK={ack})"
        except:
            return "Flow parsing error"
    
    def test_baseline_llm(self, flows_df: pd.DataFrame, model_name: str = "gpt2") -> pd.DataFrame:
        """Test baseline LLM"""
        logger.info(f"Testing Baseline LLM: {model_name}\n")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        
        results = []
        for idx, flow in flows_df.iterrows():
            try:
                true_label = str(flow['Label'])
                narrative = self.flow_to_narrative(flow)
                prompt = f"{narrative}\nBENIGN or DDoS?\nAnswer:"
                
                start = time.time()
                inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                latency = time.time() - start
                predicted = self._parse_classification(response)
                
                results.append({
                    'flow_id': idx, 'true_label': true_label, 'predicted_label': predicted,
                    'response': response[-100:], 'latency': latency, 'correct': predicted == true_label
                })
            except Exception as e:
                results.append({
                    'flow_id': idx, 'true_label': 'ERROR', 'predicted_label': 'ERROR',
                    'response': str(e)[:50], 'latency': 0, 'correct': False
                })
                logger.warning(f"Flow {idx} error: {str(e)[:50]}")
            
            if idx % 50 == 0 and idx > 0:
                logger.info(f"Processed {idx}/{len(flows_df)}")
        
        return pd.DataFrame(results)
    
    # evaluation/cicids_tester.py - with streaming detection

    def test_piperag(self, flows_df: pd.DataFrame, piperag_model) -> pd.DataFrame:
        """Test PipeRAG with streaming detection"""
        logger.info("Testing PipeRAG with streaming detection\n")
        
        results = []
        
        def detection_callback(classification, partial_text):
            """Called immediately when classification detected"""
            logger.info(f"🚨 IMMEDIATE DETECTION: {classification}")
        
        for idx, flow in flows_df.iterrows():
            try:
                true_label = str(flow['Label'])
                narrative = self.flow_to_narrative(flow)
                
                start = time.time()
                
                # Generate with streaming detection
                result = piperag_model.generate_with_streaming_retrieval(
                    narrative, 
                    max_tokens=200,
                    detection_callback=detection_callback
                )
                
                latency = time.time() - start
                predicted = piperag_model.get_simple_classification(result)
                
                results.append({
                    'flow_id': idx, 
                    'true_label': true_label, 
                    'predicted_label': predicted,
                    'response': result['full_response'][:200], 
                    'latency': latency,
                    'detection_token': result['detection_token'],  # When detected
                    'correct': predicted == true_label
                })
                
                if idx % 10 == 0 and idx > 0:
                    logger.info(f"Processed {idx}/{len(flows_df)}")
            
            except Exception as e:
                results.append({
                    'flow_id': idx, 
                    'true_label': 'ERROR', 
                    'predicted_label': 'ERROR',
                    'response': str(e)[:100], 
                    'latency': 0,
                    'detection_token': -1,
                    'correct': False
                })
                logger.warning(f"Flow {idx} error: {str(e)}")
        
        return pd.DataFrame(results)

    
    def _parse_classification(self, response: str) -> str:
        """Parse response"""
        r = response.lower()
        if 'benign' in r or 'normal' in r:
            return 'BENIGN'
        elif 'ddos' in r or 'attack' in r or 'malicious' in r:
            return 'DDoS'
        else:
            return 'UNKNOWN'
    
    def calculate_metrics(self, results_df: pd.DataFrame, model_name: str) -> Dict:
        """Calculate metrics"""
        y_true = results_df['true_label']
        y_pred = results_df['predicted_label']
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        avg_latency = results_df['latency'].mean()
        p95_latency = results_df['latency'].quantile(0.95)
        throughput = 1.0 / avg_latency if avg_latency > 0 else 0
        
        logger.info(f"\n{'='*70}\n{model_name} EVALUATION RESULTS\n{'='*70}")
        logger.info(f"Accuracy:   {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        logger.info(f"Latency:    {avg_latency:.3f}s (p95: {p95_latency:.3f}s) | Throughput: {throughput:.2f} flows/sec")
        logger.info(f"{'='*70}\n")
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1,
                'avg_latency': avg_latency, 'p95_latency': p95_latency, 'throughput': throughput}
    
    def compare_models(self, baseline_results: pd.DataFrame, piperag_results: pd.DataFrame):
        """Compare"""
        b_acc = baseline_results['correct'].mean()
        p_acc = piperag_results['correct'].mean()
        logger.info(f"\n{'='*70}\nFINAL COMPARISON\n{'='*70}")
        logger.info(f"Baseline: {b_acc:.4f} | PipeRAG: {p_acc:.4f} | Improvement: {(p_acc-b_acc)*100:+.2f}%\n")
