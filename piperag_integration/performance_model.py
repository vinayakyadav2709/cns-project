"""Abstract performance model for adaptive retrieval"""
from abc import ABC, abstractmethod

class PerformanceModel(ABC):
    @abstractmethod
    def should_trigger_retrieval(self, tokens_since_last: int) -> bool:
        pass
    
    @abstractmethod
    def predict_latency(self, vector_db_size: int) -> float:
        pass

class FixedIntervalModel(PerformanceModel):
    def __init__(self, interval=16):
        self.interval = interval
    
    def should_trigger_retrieval(self, tokens_since_last: int) -> bool:
        return tokens_since_last >= self.interval
    
    def predict_latency(self, vector_db_size: int) -> float:
        # Simple linear model
        return 0.001 * vector_db_size + 0.05  # 50ms base + 1ms per 1K vectors
