# 4) Add tiny **RAG** over local files (30–40 min)

Super-simple: embed your docs, build a FAISS index, retrieve top-k chunks, and stuff them into the prompt before
generation.

```python
# rag.py
import os, glob
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
DIM = EMB.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(DIM)

docs, chunks = [], []
for path in glob.glob("data/*.pdf"):
    text = ""
    for page in PdfReader(path).pages:
        text += page.extract_text() or ""
    for i in range(0, len(text), 800):
        chunk = text[i:i + 800]
        if chunk.strip():
            chunks.append(chunk)
            docs.append((path, i))

vecs = EMB.encode(chunks, normalize_embeddings=True)
index.add(vecs)


def retrieve(q, k=4):
    qv = EMB.encode([q], normalize_embeddings=True)
    scores, idxs = index.search(qv, k)
    return [chunks[i] for i in idxs[0]]


MODEL = "openai/gpt-oss-20b"
tok = AutoTokenizer.from_pretrained(MODEL, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype="auto", use_auth_token=True)
gen = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=400)


def answer(q):
    ctx = "\n\n".join(retrieve(q))
    prompt = f"""<|system|>
You answer strictly from the provided context. If missing, say you don't know.
<|user|>
Question: {q}

Context:
{ctx}
<|assistant|>
"""
    out = gen(prompt)[0]["generated_text"].split("<|assistant|>")[-1].strip()
    return out


if __name__ == "__main__":
    print(answer("What does the syllabus say about late submissions?"))
```
