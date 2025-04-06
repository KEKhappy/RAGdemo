from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_func():
    return HuggingFaceEmbeddings(model_name="TatonkaHF/bge-m3_en_ru")