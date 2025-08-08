from typing import List, Optional

class LLMProvider:
    def generate(self, prompt: str, *, system: Optional[str]=None, temperature: float=0.2, max_tokens: int=1024) -> str:
        raise NotImplementedError

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
