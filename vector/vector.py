

import warnings
from dotenv import load_dotenv
import os
from typing import List

# Suprimir warning deprecado de OllamaEmbeddings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant
from langchain_core.tools import Tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# ==============
# Config & setup
# ==============
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "ollama").lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Selección dinámica de embeddings
if EMBEDDINGS_PROVIDER == "openai":
    EMB = OpenAIEmbeddings()
    print("[INFO] Usando OpenAIEmbeddings para embeddings.")
elif EMBEDDINGS_PROVIDER == "gemini":
    EMB = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    print("[INFO] Usando GoogleGenerativeAIEmbeddings (Gemini) para embeddings.")
else:
    EMB = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model="nomic-embed-text")
    print(f"[INFO] Usando OllamaEmbeddings para embeddings (modelo: nomic-embed-text, url: {OLLAMA_BASE_URL}).")

def _client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_qdrant_collection(name: str) -> Qdrant:
    client = _client()
    return Qdrant(
        client=client,
        collection_name=name,
        embeddings=EMB,
    )

def get_llm():
    """
    Devuelve el LLM adecuado según LLM_PROVIDER para clasificación.
    """
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    elif LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.5-flash")
    else:
        return Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)

def classify_query_category(query: str) -> str:
    """
    Usa el LLM para mapear la consulta a una categoría del catálogo.
    """
    llm = get_llm()
    prompt = (
        "Eres un asistente que clasifica consultas de productos en categorías del catálogo. "
        "Las categorías posibles son: calzado, ropa, accesorios. "
        "Dada la consulta del usuario, responde SOLO con la categoría más relevante, sin explicaciones ni adornos. "
        "Si no corresponde a ninguna, responde SOLO con 'ninguna'.\n\n"
        f"Consulta: {query}\nCategoría:"
    )
    try:
        # Para modelos chat, usar invoke con mensaje de sistema+usuario
        if LLM_PROVIDER in ["openai", "gemini"]:
            result = llm.invoke([
                {"role": "system", "content": "Clasifica la consulta en una categoría del catálogo."},
                {"role": "user", "content": prompt}
            ])
            # El resultado puede ser un objeto Message o string
            if hasattr(result, "content"):
                category = result.content.strip().lower()
            else:
                category = str(result).strip().lower()
        else:
            # Ollama: prompt plano
            category = llm.invoke(prompt).strip().lower()
        if category in ["calzado", "ropa", "accesorios"]:
            return category
        return ""
    except Exception as e:
        print(f"[ERROR] LLM classification failed: {e}")
        return ""

# ===================
# Retrievers por KB
# ===================
def products_retriever(k: int = 5):
    vs = get_qdrant_collection("catalog_kb")
    return vs.as_retriever(search_kwargs={"k": k})

def other_retriever(k: int = 5):
    vs = get_qdrant_collection("other_kb")
    return vs.as_retriever(search_kwargs={"k": k})

# ==========================
# Funciones RAG por dominio
# ==========================
def _combine_docs_text(docs: List) -> str:
    if not docs:
        return "No se encontraron resultados relevantes."
    lines = []
    for d in docs:
        text = getattr(d, "page_content", str(d))
        meta = getattr(d, "metadata", {})
        meta = {k: v for k, v in meta.items() if k != "extras"}
        if meta:
            lines.append(f"{text}\n{meta}")
        else:
            lines.append(text)
    return "\n\n".join(lines)

def get_products_rag(query: str) -> str:
    print(f"[TOOL LOG] get_products_rag invocado con query: {query}")
    """
    Recupera información relevante del vectorstore 'catalog_kb' (productos)
    para la consulta dada y devuelve un texto combinado.
    Si el LLM puede mapear la consulta a una categoría, filtra por metadata.category.
    Si no, usa búsqueda vectorial.
    """
    category = classify_query_category(query)
    if category:
        print(f"[INFO] El LLM clasificó la consulta en la categoría: {category}")
        client = _client()
        from qdrant_client.http import models as rest
        response = client.scroll(
            collection_name="catalog_kb",
            limit=20,
            with_payload=True,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.category",
                        match=rest.MatchValue(value=category)
                    )
                ]
            )
        )
        points = response[0]
        if not points:
            return f"No se encontraron productos en la categoría '{category}'."
        class DummyDoc:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata
        docs = []
        for p in points:
            pc = p.payload.get("page_content", "")
            meta = p.payload.get("metadata", {})
            docs.append(DummyDoc(pc, meta))
        return _combine_docs_text(docs)
    # Si no, usar vector search normal
    retriever = products_retriever(k=5)
    results = retriever.invoke(query)
    return _combine_docs_text(results)

def get_other_rag(query: str) -> str:
    print(f"[TOOL LOG] get_other_rag invocado con query: {query}")
    retriever = other_retriever(k=5)
    results = retriever.invoke(query)
    return _combine_docs_text(results)

def products_tool_wrapper(query: str):
    print(f"[TOOL LOG] products_tool_wrapper query: {query}")
    return get_products_rag(query)

products_tool = Tool(
    name="products_retrieval_tool",
    func=products_tool_wrapper,
    description=(
        "Usa esta herramienta para responder preguntas sobre productos del catálogo. "
        "La entrada es una consulta en texto; la salida es un resumen concatenado "
        "de los documentos relevantes en 'catalog_kb'."
    ),
)

other_tool = Tool(
    name="other_retrieval_tool",
    func=get_other_rag,
    description=(
        "Usa esta herramienta para responder preguntas de la base 'other_kb'. "
        "La entrada es una consulta en texto; la salida es un resumen concatenado "
        "de los documentos relevantes en 'other_kb'."
    ),
)

RETRIEVAL_TOOLS = [products_tool, other_tool]
