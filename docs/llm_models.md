# What you’ll build

A tiny **local chat + RAG** app:

* Start with a baseline open model from Hugging Face.
* Swap in **gpt-oss-20b** (OpenAI’s open-weight model) via Transformers.
* Add a minimal local **RAG** pipeline over a folder of PDFs/notes.

# 0) Prereqs (15 min)

* Python 3.11+, git, and a working GPU is nice but not required (you can run smaller/quantized variants on CPU—slower).
* Create a free **Hugging Face** account and grab a token (Settings → Access Tokens).
* (Optional) If you plan to try **gpt-oss-20b** locally, budget for \~16GB VRAM/RAM territory depending on quantization
  and runtime; it’s designed to be runnable on modest hardware compared to the 120B model. ([IT Pro][1])

# 1) Project scaffold (2 min)

```bash
mkdir hf-gptoss-mini && cd hf-gptoss-mini
python -m venv .venv && source .venv/bin/activate
pip install "transformers>=4.43" accelerate bitsandbytes torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pypdf faiss-cpu sentence-transformers flask uvicorn
```

Repo shape:

```
hf-gptoss-mini/
  app.py               # REST API (Flask) or swap to FastAPI if you prefer
  chat_baseline.py     # HF baseline model chat
  chat_gptoss.py       # gpt-oss chat
  rag.py               # simple local RAG
  data/                # drop PDFs / notes here
  .env                 # HF token, etc.
```

# 2) Baseline: run any HF chat model (10–20 min)

Use a small instruct model so you know the plumbing works (e.g., `google/gemma-2-2b-it`,
`meta-llama/Llama-3.2-3B-Instruct`, etc.). This is just to prove your pipeline.

```python
# chat_baseline.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # pick a small instruct model
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype="auto"
)
pipe = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=256)


def chat(user_msg, history=None):
    prompt = f"""You are a helpful assistant.
User: {user_msg}
Assistant:"""
    out = pipe(prompt)[0]["generated_text"].split("Assistant:", 1)[-1].strip()
    return out


if __name__ == "__main__":
    print(chat("Give me 3 fun Python tips."))
```

Run:

```bash
python chat_baseline.py
```

# 3) Swap to **gpt-oss-20b** on Hugging Face (20–30 min)

OpenAI’s **gpt-oss** weights are published on Hugging Face. Pull them just like any other model with `transformers`.
Model cards + blog: install/use instructions are there. ([Hugging Face][2], [OpenAI][3])

**Note on formatting:** gpt-oss uses a specific **harmony** message format (system/assistant/user structure) in
Transformers; follow the cookbook snippet for correct prompts. ([OpenAI Cookbook][4])

```python
# chat_gptoss.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL = "openai/gpt-oss-20b"  # from HF model card
os.environ["HF_TOKEN"] = "<your_hf_token>"

tok = AutoTokenizer.from_pretrained(MODEL, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype="auto", use_auth_token=True
)

pipe = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=384)


def format_harmony(system_msg, user_msg):
    # Minimal “harmony” style wrapper; see OpenAI’s cookbook for richer tool-use format
    return f"""<|system|>
{system_msg}
<|user|>
{user_msg}
<|assistant|>
"""


def chat(user_msg):
    prompt = format_harmony("You are a concise, helpful assistant.", user_msg)
    out = pipe(prompt)[0]["generated_text"]
    return out.split("<|assistant|>")[-1].strip()


if __name__ == "__main__":
    print(chat("Explain LoRA fine-tuning in simple terms."))
```

References for the model & usage details (model card, blog, and
how-to): ([Hugging Face][2], [OpenAI][5], [OpenAI Cookbook][4])
