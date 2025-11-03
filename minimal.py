# test_piperag_fixed.py - FIXED FOR MISTRAL

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

print("\n" + "="*80)
print("PipeRAG with MISTRAL-7B (FIXED)")
print("="*80)

# ===== USE CORRECT TOKENIZER =====
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

print(f"\n[SETUP] Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
# IMPORTANT: Set pad token BEFORE model loading
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("✓ Models loaded\n")

# ===== 10 NETWORK LOGS =====
network_logs = [
    "DDoS attack: 100s duration 50 packets high rate SYN=45",
    "Normal HTTP: 30s duration 5 packets normal rate SYN=1",
    "DDoS attack: 50s duration 100 packets high rate SYN=95",
    "Normal SSH: 20s duration 3 packets slow rate SYN=1",
    "DDoS attack: 10000s duration 500 packets extreme rate SYN=480",
    "Normal HTTPS: 60s duration 8 packets normal rate SYN=1",
    "DDoS attack: 5s duration 5000 packets flood SYN=4999",
    "Normal connection: 45s duration 10 packets normal SYN=2",
    "DDoS attack: 70s duration 200 packets high rate SYN=150",
    "Normal query: 15s duration 4 packets normal SYN=1",
]

# ===== VECTOR DB =====
vector_db = {}
for i, log in enumerate(network_logs, 1):
    embedding = embedder.encode([log])[0]
    is_ddos = 'DDoS' in log
    vector_db[f"log_{i}"] = {
        'text': log,
        'embedding': embedding,
        'label': 'DDoS' if is_ddos else 'BENIGN'
    }

print("[VECTOR DB] Ready with 10 logs\n")

# ===== 10 RUNS =====
print("="*80)
print("RUNNING 10 ITERATIONS")
print("="*80)

results_summary = []

for run in range(1, 11):
    print(f"\nRUN {run}/10: {network_logs[run-1][:50]}")
    
    test_log = network_logs[run - 1]
    
    # ===== FIXED PROMPT FOR MISTRAL =====
    # Mistral expects instruction format
    prompt = f"""[INST] Classify this network flow as attack or benign:

{test_log}

Classification: [/INST]"""
    
    # Tokenize with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    generated_tokens_ids = []
    
    for token_num in range(4):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        new_token_id = outputs[0, -1].item()
        generated_tokens_ids.append(new_token_id)
        
        # Update for next iteration
        inputs = {'input_ids': outputs, 'attention_mask': torch.ones_like(outputs)}
    
    # Decode safely
    four_tokens = tokenizer.decode(generated_tokens_ids, skip_special_tokens=True).strip()
    
    # Prevent infinite loops - take only first 20 chars
    if len(four_tokens) > 20:
        four_tokens = four_tokens[:20]
    
    print(f"  4 Tokens: '{four_tokens}'")
    
    # ===== RETRIEVE =====
    tokens_embedding = embedder.encode([four_tokens])[0]
    
    similarities = {}
    for log_id, log_data in vector_db.items():
        sim = np.dot(tokens_embedding, log_data['embedding']) / (np.linalg.norm(tokens_embedding) * np.linalg.norm(log_data['embedding']))
        similarities[log_id] = (sim, log_data)
    
    sorted_results = sorted(similarities.items(), key=lambda x: x[1][0], reverse=True)
    top_log_id, (top_score, top_log_data) = sorted_results[0]
    
    # ===== DECISION =====
    if 'DDoS' in top_log_data['label'] or 'attack' in four_tokens.lower() or 'ddos' in four_tokens.lower():
        decision = "ATTACK"
    else:
        decision = "BENIGN"
    
    print(f"  Match: {top_log_id} ({top_log_data['label']}) → {decision}")
    
    results_summary.append({
        'run': run,
        'input': network_logs[run-1][:30],
        'decision': decision
    })

# ===== SUMMARY =====
print(f"\n\n" + "="*80)
print("SUMMARY")
print("="*80 + "\n")

for result in results_summary:
    print(f"Run {result['run']:2d}: {result['input']:<30} → {result['decision']}")

print("\n" + "="*80 + "\n")
