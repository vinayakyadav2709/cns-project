"""PipeRAG with PyTorch Lightning"""
import pytorch_lightning as pl
from piperag_integration.parallel_pipeline import PipeRAGParallelPipeline

class PipeRAGLightningModule(pl.LightningModule):
    def __init__(self, vector_db, model_name="microsoft/DialoGPT-medium"):
        super().__init__()
        self.pipeline = PipeRAGParallelPipeline(vector_db, model_name)
    
    def forward(self, query):
        return self.pipeline.generate_with_streaming_retrieval(query)
    
    def predict_step(self, batch, batch_idx):
        queries = batch['query']
        responses = [self.forward(q) for q in queries]
        return responses
