# piperag_integration/parallel_pipeline.py - WITH VARIABLE LOGGING

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import re
from .performance_model import FixedIntervalModel


class PipeRAGParallelPipeline:
    
    def __init__(self, vector_db, model_name="gpt2"):
        self.vector_db = vector_db
        self.model_name = model_name
        self.embedder_name = 'all-MiniLM-L6-v2'
        self.retrieval_interval = 64
        
        print("\n" + "="*70)
        print("[PipeRAG INIT] Loading models...")
        print("="*70)
        
        # LLM
        print(f"[PipeRAG INIT] Loading LLM: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        )
        print(f"[PipeRAG INIT] ✓ LLM loaded: {self.model_name}")
        
        # Embedder
        print(f"[PipeRAG INIT] Loading Embedder: {self.embedder_name}")
        self.embedder = SentenceTransformer(self.embedder_name)
        print(f"[PipeRAG INIT] ✓ Embedder loaded: {self.embedder_name}")
        
        self.retrieval_executor = ThreadPoolExecutor(max_workers=2)
        self.perf_model = FixedIntervalModel(interval=self.retrieval_interval)
        
        self.retrieval_future = None
        self.detection_callback = None
        
        vector_db_type = type(vector_db).__name__
        print(f"[PipeRAG INIT] ✓ Retrieval interval: {self.retrieval_interval} tokens")
        print(f"[PipeRAG INIT] ✓ Vector DB: {vector_db_type}")
        print("="*70)
        print("[PipeRAG INIT] Ready!\n")
    
    def generate_with_streaming_retrieval(self, query, max_tokens=200, 
                                         detection_callback=None):
        """Generate with detailed logging"""
        
        self.detection_callback = detection_callback
        
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " "*20 + "NEW FLOW ANALYSIS" + " "*31 + "║")
        print("╚" + "="*68 + "╝")
        
        # ===== LOG INPUT =====
        print("\n[INPUT] Flow Data:")
        print("-" * 70)
        print(query)
        print("-" * 70)
        
        # ===== BUILD PROMPT =====
        prompt = f"""Analyze this network flow and output in EXACT format:

<FLOW>
{query}
</FLOW>

<ANALYSIS>
Step 1: Examine flow characteristics
Step 2: Identify patterns
</ANALYSIS>

<CLASSIFICATION>BENIGN</CLASSIFICATION>
or
<CLASSIFICATION>ATTACK:DDoS</CLASSIFICATION>

<CONFIDENCE>HIGH/MEDIUM/LOW</CONFIDENCE>

Now analyze:
<FLOW>{query}</FLOW>
<ANALYSIS>
"""
        
        # ===== LOG PROMPT =====
        print("\n[PROMPT] Sent to LLM:")
        print("-" * 70)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("-" * 70)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        generated_tokens = []
        tokens_since_retrieval = 0
        retrieval_injected = False
        detected_classification = None
        detection_triggered = False
        
        first_64_tokens = []
        
        print(f"\n[GENERATION] Starting token-by-token generation (max {max_tokens})...")
        print(f"[GENERATION] Using model: {self.model_name}")
        print(f"[GENERATION] Retrieval will trigger at token {self.retrieval_interval}")
        
        # ===== TOKEN-BY-TOKEN GENERATION =====
        for i in range(max_tokens):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            new_token = outputs[0, -1].item()
            generated_tokens.append(new_token)
            tokens_since_retrieval += 1
            
            # Store first 64 tokens
            if i < 64:
                token_text = self.tokenizer.decode([new_token], skip_special_tokens=True)
                first_64_tokens.append(token_text)
            
            current_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # ===== STREAMING DETECTION =====
            if not detection_triggered:
                classification = self._check_classification_tag(current_text)
                if classification:
                    detected_classification = classification
                    detection_triggered = True
                    
                    print(f"\n[DETECTION] 🚨 Token {i}: Classification detected!")
                    print(f"[DETECTION] Classification: {classification}")
                    print(f"[DETECTION] Partial response at detection:")
                    print("-" * 70)
                    print(current_text[:300])
                    print("-" * 70)
                    
                    if self.detection_callback:
                        self.detection_callback(classification, current_text)
            
            # ===== PARALLEL RETRIEVAL TRIGGER =====
            if self.perf_model.should_trigger_retrieval(tokens_since_retrieval) and not retrieval_injected:
                print(f"\n[RETRIEVAL] 🔥 Token {i}: Triggering retrieval")
                
                # Log what we're using for retrieval
                retrieval_query = current_text
                print(f"[RETRIEVAL] Query text for retrieval (first 200 chars):")
                print("-" * 70)
                print(retrieval_query[:200])
                print("-" * 70)
                print(f"[RETRIEVAL] Using embedder: {self.embedder_name}")
                print(f"[RETRIEVAL] Vector DB: {type(self.vector_db).__name__}")
                
                self.retrieval_future = self.retrieval_executor.submit(
                    self._async_retrieve,
                    retrieval_query
                )
                
                tokens_since_retrieval = 0
                print(f"[RETRIEVAL] ⚡ Retrieval running in background thread...")
            
            # ===== INJECT CONTEXT WHEN READY =====
            if self.retrieval_future is not None and self.retrieval_future.done() and not retrieval_injected:
                print(f"\n[CONTEXT] ✅ Token {i}: Retrieval completed, injecting context")
                
                retrieved_context = self.retrieval_future.result()
                
                print(f"[CONTEXT] Retrieved patterns:")
                print("-" * 70)
                print(retrieved_context[:400])
                print("-" * 70)
                
                prompt_with_context = f"""{prompt}
{current_text}

<RETRIEVED_PATTERNS>
{retrieved_context}
</RETRIEVED_PATTERNS>

Continue analysis considering retrieved patterns:
"""
                
                inputs = self.tokenizer(prompt_with_context, return_tensors="pt", 
                                       truncation=True, max_length=512)
                
                retrieval_injected = True
                print(f"[CONTEXT] 🎯 Context injected, continuing generation...")
            else:
                inputs = {'input_ids': outputs}
            
            if new_token == self.tokenizer.eos_token_id:
                print(f"\n[GENERATION] EOS token reached at token {i}")
                break
        
        # ===== LOG FIRST 64 TOKENS =====
        print(f"\n[TOKENS] First 64 tokens generated:")
        print("-" * 70)
        first_64_text = "".join(first_64_tokens)
        print(first_64_text)
        print("-" * 70)
        
        # ===== FINAL RESPONSE =====
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"\n[RESULT] Final classification: {detected_classification or 'UNKNOWN'}")
        print(f"[RESULT] Total tokens generated: {len(generated_tokens)}")
        print(f"[RESULT] Detection triggered at token: {i if detection_triggered else -1}")
        
        return {
            'classification': detected_classification or 'UNKNOWN',
            'full_response': response,
            'detection_token': i if detection_triggered else -1,
            'first_64_tokens': first_64_text
        }
    
    def _check_classification_tag(self, text):
        """Parse classification tag"""
        match = re.search(r'<CLASSIFICATION>(.*?)</CLASSIFICATION>', text, re.IGNORECASE)
        
        if match:
            classification = match.group(1).strip()
            return classification
        
        if '<CLASSIFICATION>' in text:
            partial = text.split('<CLASSIFICATION>')[-1]
            
            if 'BENIGN' in partial.upper():
                return 'BENIGN'
            elif 'ATTACK:DDOS' in partial.upper():
                return 'ATTACK:DDoS'
            elif 'ATTACK:PORTSCAN' in partial.upper():
                return 'ATTACK:PortScan'
            elif 'ATTACK' in partial.upper():
                return 'ATTACK:Unknown'
        
        return None
    
    def _async_retrieve(self, query_text):
        """Async retrieval with logging"""
        start = time.time()
        print(f"[Retrieval Thread] Starting retrieval...")
        print(f"[Retrieval Thread] Using embedder: {self.embedder_name}")
        
        query_embedding = self._get_query_embedding(query_text)
        print(f"[Retrieval Thread] Embedding shape: {query_embedding.shape}")
        
        retrieved = self.vector_db.retrieve(query_embedding, top_k=5)
        print(f"[Retrieval Thread] Retrieved {len(retrieved)} patterns")
        
        context_lines = []
        for idx, (score, text, metadata) in enumerate(retrieved, 1):
            label = metadata.get('label', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
            pattern_line = f"Pattern {idx} (score={score:.3f}, label={label}): {text[:150]}"
            context_lines.append(pattern_line)
            print(f"[Retrieval Thread] {pattern_line}")
        
        context = "\n".join(context_lines)
        
        elapsed = time.time() - start
        print(f"[Retrieval Thread] ✅ Done in {elapsed:.2f}s")
        
        return context
    
    def _get_query_embedding(self, query):
        """Get embedding"""
        semantic = self.embedder.encode([query])[0]
        padding = np.zeros(17)
        embedding = np.concatenate([semantic, padding])
        return embedding
    
    def get_simple_classification(self, result):
        """Extract simple label"""
        classification = result['classification']
        
        if 'BENIGN' in classification:
            return 'BENIGN'
        elif 'DDOS' in classification.upper():
            return 'DDoS'
        elif 'ATTACK' in classification:
            return 'ATTACK'
        else:
            return 'UNKNOWN'
