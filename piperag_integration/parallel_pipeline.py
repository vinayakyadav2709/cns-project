# piperag_integration/parallel_pipeline.py - WITH STREAMING DETECTION

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
    """
    PipeRAG with STREAMING ANOMALY DETECTION
    Detects anomalies AS SOON AS classification tag is generated
    """
    
    def __init__(self, vector_db, model_name="gpt2"):
        self.vector_db = vector_db
        
        print("[PipeRAG] Loading models...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        
        print("[PipeRAG] Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.retrieval_executor = ThreadPoolExecutor(max_workers=2)
        self.perf_model = FixedIntervalModel(interval=64)
        
        self.retrieval_future = None
        self.lock = threading.Lock()
        
        # Callback for immediate detection
        self.detection_callback = None
        
        print("[PipeRAG] Ready!")
    
    def generate_with_streaming_retrieval(self, query, max_tokens=200, 
                                         detection_callback=None):
        """
        Generate with STREAMING detection
        Calls detection_callback IMMEDIATELY when classification tag detected
        """
        
        self.detection_callback = detection_callback
        
        # ===== PROMPT: Force LLM to use structured format =====
        prompt = f"""Analyze this network flow and output in EXACT format:

<FLOW>
{query}
</FLOW>

<ANALYSIS>
Step 1: Examine characteristics
[Your reasoning here]
Step 2: Compare patterns
[Your analysis here]
</ANALYSIS>

<CLASSIFICATION>BENIGN</CLASSIFICATION>
or
<CLASSIFICATION>ATTACK:DDoS</CLASSIFICATION>
or
<CLASSIFICATION>ATTACK:PortScan</CLASSIFICATION>

<CONFIDENCE>HIGH/MEDIUM/LOW</CONFIDENCE>

Now analyze:
<FLOW>{query}</FLOW>

<ANALYSIS>
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        generated_tokens = []
        tokens_since_retrieval = 0
        retrieval_injected = False
        
        # Detection state
        detected_classification = None
        detection_triggered = False
        
        print(f"[PipeRAG] Starting streaming generation...")
        
        # ===== TOKEN-BY-TOKEN GENERATION =====
        for i in range(max_tokens):
            # Generate next token
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
            
            # Decode current generation
            current_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # ===== STREAMING DETECTION: Check for classification tag =====
            if not detection_triggered:
                classification = self._check_classification_tag(current_text)
                if classification:
                    detected_classification = classification
                    detection_triggered = True
                    
                    # TRIGGER IMMEDIATE CALLBACK
                    print(f"[PipeRAG] 🚨 DETECTION at token {i}: {classification}")
                    
                    if self.detection_callback:
                        self.detection_callback(classification, current_text)
                    
                    # For evaluation, we can stop here or continue for full analysis
                    # For real-time: STOP and return immediately
                    # For evaluation: continue to get full reasoning
            
            # ===== PARALLEL RETRIEVAL TRIGGER =====
            if self.perf_model.should_trigger_retrieval(tokens_since_retrieval) and not retrieval_injected:
                print(f"[PipeRAG] 🔥 Token {i}: Triggering retrieval")
                
                self.retrieval_future = self.retrieval_executor.submit(
                    self._async_retrieve,
                    current_text
                )
                
                tokens_since_retrieval = 0
                print(f"[PipeRAG] ⚡ Retrieval running in parallel...")
            
            # ===== INJECT CONTEXT WHEN READY =====
            if self.retrieval_future is not None and self.retrieval_future.done() and not retrieval_injected:
                print(f"[PipeRAG] ✅ Token {i}: Context injection")
                
                retrieved_context = self.retrieval_future.result()
                
                prompt_with_context = f"""{prompt}
{current_text}

<RETRIEVED_PATTERNS>
{retrieved_context}
</RETRIEVED_PATTERNS>

Continue analysis:
"""
                
                inputs = self.tokenizer(prompt_with_context, return_tensors="pt", 
                                       truncation=True, max_length=512)
                
                retrieval_injected = True
            else:
                inputs = {'input_ids': outputs}
            
            # Stop at EOS or after classification
            if new_token == self.tokenizer.eos_token_id:
                break
        
        # Final response
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            'classification': detected_classification or 'UNKNOWN',
            'full_response': response,
            'detection_token': i if detection_triggered else -1
        }
    
    def _check_classification_tag(self, text):
        """
        Parse classification tag AS SOON AS it appears
        Returns: 'BENIGN', 'ATTACK:DDoS', 'ATTACK:PortScan', etc.
        """
        
        # Look for <CLASSIFICATION>XXX</CLASSIFICATION> tag
        match = re.search(r'<CLASSIFICATION>(.*?)</CLASSIFICATION>', text, re.IGNORECASE)
        
        if match:
            classification = match.group(1).strip()
            return classification
        
        # Fallback: look for keywords if tag not complete yet
        if '<CLASSIFICATION>' in text:
            # Tag started, check partial content
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
        """Async retrieval in parallel"""
        start = time.time()
        print(f"[Retrieval] Starting...")
        
        query_embedding = self._get_query_embedding(query_text)
        retrieved = self.vector_db.retrieve(query_embedding, top_k=5)
        
        context_lines = []
        for idx, (score, text, metadata) in enumerate(retrieved, 1):
            label = metadata.get('label', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
            context_lines.append(f"Pattern {idx} (score={score:.2f}, label={label}): {text[:150]}")
        
        context = "\n".join(context_lines)
        
        elapsed = time.time() - start
        print(f"[Retrieval] ✅ Done in {elapsed:.2f}s")
        
        return context
    
    def _get_query_embedding(self, query):
        """Get embedding"""
        semantic = self.embedder.encode([query])[0]
        padding = np.zeros(17)
        return np.concatenate([semantic, padding])
    
    def get_simple_classification(self, result):
        """Extract simple label for evaluation"""
        classification = result['classification']
        
        if 'BENIGN' in classification:
            return 'BENIGN'
        elif 'DDOS' in classification.upper():
            return 'DDoS'
        elif 'ATTACK' in classification:
            return 'ATTACK'
        else:
            return 'UNKNOWN'
