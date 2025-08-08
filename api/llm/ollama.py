import os, httpx, json
from typing import List, Optional
from .base import LLMProvider

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, gen_model: str, embed_model: str):
        self.base_url = base_url.rstrip('/')
        self.gen_model = gen_model
        self.embed_model = embed_model

    def generate(self, prompt: str, *, system: Optional[str]=None, temperature: float=0.2, max_tokens: int=1024) -> str:
        payload = {
            "model": self.gen_model,
            "prompt": (f"<<SYS>>{system}<</SYS>>\n" if system else "") + prompt,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False
        }
        with httpx.Client(timeout=600) as client:
            r = client.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "")

    def embed(self, texts: List[str]) -> List[List[float]]:
        with httpx.Client(timeout=600) as client:
            r = client.post(f"{self.base_url}/api/embeddings", json={"model": self.embed_model, "input": texts})
            r.raise_for_status()
            data = r.json()
            return data.get("embeddings", [])
