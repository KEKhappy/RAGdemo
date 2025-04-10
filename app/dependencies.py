from functools import lru_cache

from fastapi import Depends
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from llama_cpp import Llama
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
from app.core.rag_service import RAGService
from app.services.db_service import VectorDB
from app.services.llm_service import LLMService

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME
    )

@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()

@lru_cache(maxsize=1)
def get_vector_db() -> Chroma:
    return Chroma(
        persist_directory=settings.CHROMA_PATH,
        embedding_function=get_embedding_model()
    )

def get_document_loader() -> PyPDFDirectoryLoader:
    return PyPDFDirectoryLoader(settings.DOCUMENTS_PATH)

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.SPLITTER.CHUNK_SIZE,
        chunk_overlap=settings.SPLITTER.CHUNK_OVERLAP
    )

@lru_cache(maxsize=1)
def get_rag_service(
    vector_db: VectorDB = Depends(get_vector_db),
    llm_service: LLMService = Depends(get_llm_service)
) -> RAGService:
    return RAGService(vector_db, llm_service)