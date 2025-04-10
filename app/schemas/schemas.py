from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["Что такое ИК датчик?", "Какие степени защиты есть у ИК датчиков?"],
        description="Вопрос пользователя для RAG-системы"
    )

class AnswerResponse(BaseModel):
    answer: str = Field(
        ...,
        examples=["ИК датчики это ... Sources: data:1:1, ...","Степени защиты ИК датчиков: 1)... Sources: data:1:0, ..."],
        description="Сгенерированный ответ LLM и список источников"
    )

class ErrorResponse(BaseModel):
    error: str = Field(
        ...,
        examples=["Some error type"],
        description="Тип ошибки"
    )
    details: str = Field(
        ...,
        examples=["Error description"],
        description="Детализированное описание ошибки"
    )