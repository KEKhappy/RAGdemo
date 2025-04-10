from langchain.schema.document import Document
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings

class DocumentService:
    def __init__(self):
        self.loader = PyPDFDirectoryLoader(settings.DOCUMENTS_PATH)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.SPLITTER.CHUNK_SIZE,
            chunk_overlap=settings.SPLITTER.CHUNK_OVERLAP
        )

    def load_and_split(self) -> list[Document]:
        docs = self.loader.load()
        return self.splitter.split_documents(docs)