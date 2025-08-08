Perfect — a **model swap harness** will make it trivial to switch between **Llama 3** and **GPT-OSS** in your Dev
Assistant without touching the RAG or UI code.

Here’s the plan:

---

## **1. Concept**

* **Single provider interface** → any model backend just implements `generate()` and `embed()`.
* Configurable via `.env` (e.g., `LLM_PROVIDER=llama3` or `gpt-oss`).
* Same chunking, embedding, and RAG logic — only the model-specific wrapper changes.
* Lets you run **A/B tests** by flipping env vars, no restart of the DB or frontend needed.

---

## **2. API Layer Change**

**`llm/base.py`**

```python
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass
```

**`llm/llama3.py`**

```python
import requests, os


class Llama3Provider:
    def __init__(self):
        self.model = os.getenv("LLM_MODEL", "llama3")

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt}
        )
        return resp.json()["response"]

    def embed(self, text: str) -> list[float]:
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "input": text}
        )
        return resp.json()["embedding"]
```

**`llm/gpt_oss.py`**

```python
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer


class GPTOSSProvider:
    def __init__(self):
        model_id = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto",
                                                          use_auth_token=True)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=400)
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def generate(self, prompt: str) -> str:
        harmony_prompt = f"<|system|>\nYou are a concise, helpful assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"
        return self.pipe(harmony_prompt)[0]["generated_text"].split("<|assistant|>")[-1].strip()

    def embed(self, text: str) -> list[float]:
        return self.embed_model.encode(text, normalize_embeddings=True).tolist()
```

**`llm/provider.py`**

```python
import os
from .llama3 import Llama3Provider
from .gpt_oss import GPTOSSProvider


def get_provider():
    choice = os.getenv("LLM_PROVIDER", "llama3").lower()
    if choice == "llama3":
        return Llama3Provider()
    elif choice == "gpt-oss":
        return GPTOSSProvider()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {choice}")
```

---

## **3. Usage in RAG**

```python
from llm.provider import get_provider

llm = get_provider()


def answer(query: str) -> str:
    context = retrieve_chunks(query)
    prompt = f"Question: {query}\n\nContext:\n{context}\n"
    return llm.generate(prompt)
```

---

## **4. `.env` toggle**

```bash
# Use Llama 3 locally
LLM_PROVIDER=llama3
LLM_MODEL=llama3

# Or swap to GPT-OSS
LLM_PROVIDER=gpt-oss
LLM_MODEL=openai/gpt-oss-20b
```

---

## **5. Benefits**

* **Hot-swappable** models — same RAG pipeline, same embedding interface.
* Run **side-by-side comparisons** for the same queries.
* Add **more providers later** (Claude, Mistral, etc.) by just adding a new `*.py` file.

---

If you want, I can also give you:

* **A/B testing harness** → sends the same query to both models and logs differences.
* **Eval dashboard** → scores model outputs for accuracy, completeness, and style using your own notes as ground truth.

That would turn your project into a *hands-on model lab* while you build the assistant.
