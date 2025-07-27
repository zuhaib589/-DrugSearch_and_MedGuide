import os, torch, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --------------------
# Load embedding model (for RAG retrieval)
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    embedder = None
    print("❌ Failed to load sentence transformer:", e)

# --------------------
# Load a lightweight open LLM
llm_id = "StabilityAI/StableLM-Zephyr-3B"
try:
    tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(llm_id, device_map="auto")
    llm_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
except Exception as e:
    llm_pipe = None
    print("❌ Failed to load LLM model:", e)

# --------------------
def build_knowledge_base(folder):
    if not embedder:
        return None, [], []

    texts, sources = [], []
    for fn in os.listdir(folder):
        if fn.endswith(".txt"):
            with open(os.path.join(folder, fn), encoding="utf8") as f:
                texts.append(f.read())
                sources.append(fn)

    embeddings = embedder.encode(texts, convert_to_tensor=True)
    return embeddings, texts, sources

def get_medical_guidance_with_llm(query, kb_embeddings=None, kb_texts=None, threshold=0.5):
    if not llm_pipe:
        return "LLM model not available."

    if kb_embeddings is not None and kb_texts is not None and embedder:
        q_emb = embedder.encode([query], convert_to_tensor=True)
        sims = cosine_similarity(q_emb.cpu().numpy(), kb_embeddings.cpu().numpy())[0]
        idx = int(np.argmax(sims))
        if sims[idx] >= threshold:
            context = kb_texts[idx]
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            resp = llm_pipe(prompt)[0]["generated_text"]
            return resp.split("Answer:")[-1].strip()

    # Fallback if no KB match
    prompt = f"Question: {query}\nAnswer:"
    resp = llm_pipe(prompt)[0]["generated_text"]
    return resp.split("Answer:")[-1].strip()
