from pydantic_settings import BaseSettings
from huggingface_hub import hf_hub_download

PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ты — ассистент, отвечающий строго на основе предоставленного контекста. Правила:
1.Отвечай только на русском языке
2.Используй исключительно факты из контекста
3.Ответ должен содержать 1-5 предложений
4.Сохраняй техническую точность<|eot_id|><|start_header_id|>user<|end_header_id|>
Контекст:
{context}

Вопрос: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


class LLMSettings(BaseSettings):
    LLM_REPO_ID: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    LLM_FILENAME: str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    MODEL_CACHE_DIR: str = "./models"
    #Случайность вывода
    TEMPERATURE: float = 0.7
    #Макс вывод
    MAX_TOKENS: int = 128
    #Логи модели
    VERBOSE: bool = False
    #Контекст
    N_CTX: int = 4096
    #Стоп токен
    STOP: list[str] = ["<|eot_id|>"]

    @property
    def MODEL_PATH(self) -> str:
        return hf_hub_download(
            repo_id=self.LLM_REPO_ID,
            filename=self.LLM_FILENAME,
            cache_dir=self.MODEL_CACHE_DIR
        )

class TextSplitterSettings(BaseSettings):
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 80

class Settings(BaseSettings):
    LLM: LLMSettings = LLMSettings()
    SPLITTER: TextSplitterSettings = TextSplitterSettings()

    PROMPT_TEMPLATE: str = PROMPT

    EMBEDDING_MODEL_NAME: str = "TatonkaHF/bge-m3_en_ru"
    CHROMA_PATH: str = "chroma"
    DOCUMENTS_PATH: str = "data"

    RESET: bool = False

settings = Settings()