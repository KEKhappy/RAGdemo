from langchain_chroma import Chroma
from app.utils.doc_util import calculate_chunk_ids
from app.config import settings
from langchain.schema.document import Document
import os
import shutil

class VectorDB:
    def __init__(self, embedding_function):
        self.db = Chroma(
            persist_directory=settings.CHROMA_PATH,
            embedding_function=embedding_function
        )
    def add_documents(self, chunks: list[Document]):
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Проверка БД
        existing_items = self.db.get(include=[])
        existing_ids = set(existing_items["ids"])

        # Поиск несуществующих документов
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        # Добавление чанков
        if len(new_chunks):
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.db.add_documents(new_chunks, ids=new_chunk_ids)

        return len(new_chunks)

    def similarity_search(self, query: str, k: int = 5):
        return self.db.similarity_search_with_score(query, k=k)

    def clear(self):
        if os.path.exists(settings.CHROMA_PATH):
            shutil.rmtree(settings.CHROMA_PATH)