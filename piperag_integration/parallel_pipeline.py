import torch
import numpy as np  # ADD THIS LINE
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from .performance_model import FixedIntervalModel  # CHANGE THIS


class PipeRAGParallelPipeline:
    def __init__(self, vector_db, model_name="gpt2-large"):
        self.vector_db = vector_db
        
        print("[PipeRAG] Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        
        # Parallel executors
        self.retrieval_executor = ThreadPoolExecutor(max_workers=2)
        self.generation_executor = ThreadPoolExecutor(max_workers=1)
        
        # Performance model
        self.perf_model = FixedIntervalModel(interval=16)
        
        # Shared state
        self.retrieval_cache = {}
        self.lock = threading.Lock()
        
        print("[PipeRAG] Ready!")
    
    def generate_with_streaming_retrieval(self, query, max_tokens=100):
        """Generate with parallel retrieval at fixed intervals"""
        print(f"[PipeRAG] Generating for query: {query[:50]}...")
        
        # Initial retrieval
        query_embedding = self._get_query_embedding(query)
        retrieved = self.vector_db.retrieve(query_embedding, top_k=5)
        context = "\n".join([text for _, text, _ in retrieved])
        
        prompt = f"Context:\n{context}\n\nQuery: {query}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        
        generated_tokens = []
        tokens_since_retrieval = 0
        
        # Token-by-token generation with periodic retrieval
        for i in range(max_tokens):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            new_token = outputs[0, -1].item()
            generated_tokens.append(new_token)
            tokens_since_retrieval += 1
            
            # Check if retrieval needed
            if self.perf_model.should_trigger_retrieval(tokens_since_retrieval):
                # Async retrieval in background
                future = self.retrieval_executor.submit(
                    self.vector_db.retrieve, query_embedding, 3
                )
                tokens_since_retrieval = 0
            
            # Update inputs for next token
            inputs = {'input_ids': outputs}
            
            if new_token == self.tokenizer.eos_token_id:
                break
        
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response
    
    def _get_query_embedding(self, query):
        """Get embedding for query (simplified)"""
        # Use first 384 dims as semantic embedding
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        semantic = embedder.encode([query])[0]
        
        # Pad to match session embedding dim (401)
        padding = np.zeros(17)
        return np.concatenate([semantic, padding])
