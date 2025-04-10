from app.services.db_service import VectorDB
from app.services.llm_service import LLMService
from app.config import settings


class RAGService:
    def __init__(self, vector_db: VectorDB, llm_service: LLMService):
        self.vector_db = vector_db
        self.llm_service = llm_service
        self.prompt_template = settings.PROMPT_TEMPLATE

    def query(self, question: str) -> str:
        # Поиск в векторной БД
        results = self.vector_db.similarity_search(question, k=5)

        # Формирование контекста
        context = "\n\n---\n\n".join([doc.page_content for doc in results])

        print("Context generated")

        # Генерация промпта
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )

        print("Prompt generated")

        # Генерация ответа
        response = self.llm_service.generate_response(prompt)
        response_text = response["choices"][0]["text"]

        print("Answer generated")

        # Добавляем к выводу номера использованных документов и их чанков
        sources = [doc.metadata.get("id", None) for doc in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"

        print("Source generated :\n" + formatted_response)

        return formatted_response