"""Anomaly detection from LLM outputs"""
import re
from typing import List, Dict

class AnomalyDetector:
    def __init__(self):
        self.attack_keywords = [
            'ddos', 'dos', 'hulk', 'goldeneye', 'slowloris',
            'attack', 'malicious', 'suspicious', 'intrusion'
        ]
    
    def detect_from_response(self, response: str, ground_truth_label: str) -> Dict:
        """Extract attack classification from LLM response"""
        response_lower = response.lower()
        
        # Check if any attack keyword mentioned
        detected_attack = any(kw in response_lower for kw in self.attack_keywords)
        
        # Compare with ground truth
        is_attack_gt = ground_truth_label.upper() != 'BENIGN'
        
        return {
            'predicted_attack': detected_attack,
            'ground_truth_attack': is_attack_gt,
            'correct': detected_attack == is_attack_gt,
            'response': response
        }
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute precision, recall, F1"""
        tp = sum(1 for r in results if r['predicted_attack'] and r['ground_truth_attack'])
        fp = sum(1 for r in results if r['predicted_attack'] and not r['ground_truth_attack'])
        fn = sum(1 for r in results if not r['predicted_attack'] and r['ground_truth_attack'])
        tn = sum(1 for r in results if not r['predicted_attack'] and not r['ground_truth_attack'])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'total_detected': tp + fp,
            'total_actual': tp + fn
        }
