import os, torch, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --------------------
# Load embedding model (for RAG retrieval)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------
# Load a lightweight, open LLM
llm_id = "StabilityAI/StableLM-Zephyr-3B"  # Instruction fine‑tuned
tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(llm_id,device_map="auto")
llm_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

# --------------------
def build_knowledge_base(folder):
    texts, sources = [], []
    for fn in os.listdir(folder):
        if fn.endswith(".txt"):
            txt = open(os.path.join(folder, fn), encoding="utf8").read()
            texts.append(txt)
            sources.append(fn)
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    return embeddings, texts, sources

def get_medical_guidance_with_llm(query, kb_embeddings=None, kb_texts=None, threshold=0.5):
    # RAG retrieval
    if kb_embeddings is not None and kb_texts is not None:
        q_emb = embedder.encode([query], convert_to_tensor=True)
        sims = cosine_similarity(q_emb.cpu().numpy(), kb_embeddings.cpu().numpy())[0]
        idx = int(np.argmax(sims))
        if sims[idx] >= threshold:
            context = kb_texts[idx]
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            resp = llm_pipe(prompt)[0]["generated_text"]
            return resp.split("Answer:")[-1].strip()
    # Fallback LLM only
    prompt = f"Question: {query}\nAnswer:"
    resp = llm_pipe(prompt)[0]["generated_text"]
    return resp.split("Answer:")[-1].strip()