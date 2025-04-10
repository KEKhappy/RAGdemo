from llama_cpp import Llama
from app.config import settings

class LLMService:
    def __init__(self):
        self.llm = Llama(
            model_path=settings.LLM.MODEL_PATH,
            n_ctx=settings.LLM.N_CTX,
            verbose=settings.LLM.VERBOSE
        )

    def generate_response(self, prompt_with_question: str):
        return self.llm.create_completion(
            prompt=prompt_with_question,
            max_tokens=settings.LLM.MAX_TOKENS,
            temperature=settings.LLM.TEMPERATURE,
            stop=settings.LLM.STOP
        )