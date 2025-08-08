from settings import Settings
from .ollama import OllamaProvider

def get_llm(settings: Settings):
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings.ollama_url, settings.gen_model, settings.embed_model)
    # You can extend with vLLM/TGI providers later while keeping this interface.
    return OllamaProvider(settings.ollama_url, settings.gen_model, settings.embed_model)
