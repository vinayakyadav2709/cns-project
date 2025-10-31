"""Baseline - Gemma LLM with chunked context"""
import pytorch_lightning as pl
from transformers import pipeline

class BaselineLongContextModel(pl.LightningModule):
    def __init__(self, model_name="google/gemma-2b"):
        super().__init__()
        self.llm = pipeline("text-generation", model=model_name, device=-1, torch_dtype="float32")
    
    def generate_full_context(self, logs_text, query, max_length=512):
        """Gemma analyzes chunked logs"""
        chunk_size = 500
        chunks = [logs_text[i:i+chunk_size] for i in range(0, len(logs_text), chunk_size)][:3]
        
        all_chunks = "\n---CHUNK---\n".join(chunks)
        prompt = f"""Analyze network logs for attacks:

LOGS:
{all_chunks}

Question: Are there any DDoS or attack patterns?
Answer (MALICIOUS or BENIGN):"""
        
        output = self.llm(prompt, max_length=20, do_sample=False)
        answer = output[0]['generated_text']
        
        if "malicious" in answer.lower() or "attack" in answer.lower() or "ddos" in answer.lower():
            return "MALICIOUS"
        return "BENIGN"
