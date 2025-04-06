import argparse
from huggingface_hub import hf_hub_download
from langchain.prompts import ChatPromptTemplate
from get_embedding_func import get_embedding_func
from langchain_chroma import Chroma
from llama_cpp import Llama

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ты — ассистент, отвечающий строго на основе предоставленного контекста. Правила:
1.Отвечай только на русском языке
2.Используй исключительно факты из контекста
3.Ответ должен содержать 1-5 предложений
4.Сохраняй техническую точность<|eot_id|><|start_header_id|>user<|end_header_id|>
Контекст:
{context}

Вопрос: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


model_path = hf_hub_download(
    #Llama с квантованием 4 bit
    repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    cache_dir="./models"
)
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_gpu_layers=0,
    verbose=False,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    #Подготовка БД
    embedding_function = get_embedding_func()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    #Поиск в БД (5 ближайших результатов)
    results = db.similarity_search_with_score(query_text, k=5)

    #Подготовка промпта
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results]) #Разделение результатов ---
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print(prompt)

    #Получаем ответ от LLm
    response = llm.create_completion(
        prompt=prompt,
        max_tokens=128,
        temperature=0.7,
        stop=["<|eot_id|>"]
    )
    response_text = response["choices"][0]["text"]

    #Добавляем к выводу номера использованных документов и их чанков
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()