from app.services.doc_service import DocumentService
from app.services.db_service import VectorDB
from app.dependencies import get_embedding_model

def initialize_database(reset: bool = False):
    embedding_function = get_embedding_model()
    vector_db = VectorDB(embedding_function)

    #Очищаем если нужно
    if reset:
        vector_db.clear()

    #Добавляем документы (если есть)
    document_service = DocumentService()
    chunks = document_service.load_and_split()
    added_count = vector_db.add_documents(chunks)

    #Выводим результат
    if added_count > 0:
        print(f"Добавлено {added_count} чанков")
    else:
        print("Добавить в БД нечего")