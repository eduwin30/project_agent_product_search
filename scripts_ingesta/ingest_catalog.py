import os, csv, sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

# Cargar el .env correcto siempre
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
print(f"[INFO] Usando Qdrant en: {QDRANT_URL}")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

# Selección dinámica de embeddings
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "ollama").lower()
if EMBEDDINGS_PROVIDER == "openai":
    EMB = OpenAIEmbeddings()
    print("[INFO] Usando OpenAIEmbeddings para embeddings.")
elif EMBEDDINGS_PROVIDER == "gemini":
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("[ERROR] Falta GOOGLE_API_KEY en el entorno para usar GeminiEmbeddings.", file=sys.stderr)
        sys.exit(1)
    EMB = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    print("[INFO] Usando GoogleGenerativeAIEmbeddings (Gemini) para embeddings.")
else:
    # Por defecto usa Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
    EMB = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    print(f"[INFO] Usando OllamaEmbeddings para embeddings (modelo: {OLLAMA_MODEL}, url: {OLLAMA_BASE_URL}).")

def normalize_headers(row: dict) -> dict:
    # Limpieza y normalización de cabeceras
    return { (k or "").strip().lower().replace("\ufeff", ""): (v or "").strip() for k, v in row.items() }

def run(path=None, collection="catalog_kb"):
    # Usar path absoluto por defecto relativo a este script
    if path is None:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'catalog_samples.csv'))
    docs = []

    if not os.path.exists(path):
        print(f"[ERROR] No se encontró el archivo: {path}", file=sys.stderr)
        sys.exit(1)

    # Abrimos con newline="" y utf-8-sig para soportar BOM y CRLF
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        # Normalizamos líneas de salto
        content = f.read().replace("\r\n", "\n").strip()
        lines = content.split("\n")

        # Si no hay encabezados detectables
        if not lines or len(lines) < 2:
            print("[ERROR] El archivo CSV no contiene datos.")
            sys.exit(1)

        # Volvemos a procesar como CSV
        reader = csv.DictReader(lines)
        print("[INFO] Cabeceras detectadas:", reader.fieldnames)

        for raw in reader:
            if not raw:
                continue
            row = normalize_headers(raw)
            title  = row.get("title")    or row.get("nombre") or ""
            brand  = row.get("brand")    or row.get("marca")  or ""
            price  = row.get("price")    or row.get("precio") or ""
            cat    = row.get("category") or row.get("categoria") or ""
            sku    = row.get("sku") or ""

            if not title:
                print(f"[WARN] Fila omitida por falta de 'title': {row}")
                continue

            text = f"{title} — {brand} — USD {price} — {cat}".strip(" —")
            meta = {"sku": sku, "brand": brand, "price": price, "category": cat}
            docs.append(Document(page_content=text, metadata=meta))

    if not docs:
        print("[WARN] No se generaron documentos. Revisa cabeceras y contenido del CSV.")
        sys.exit(1)

    # Insertar documentos en Qdrant usando la nueva API
    Qdrant.from_documents(
        documents=docs,
        embedding=EMB,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection,
    )

    print(f"✅ Ingestados {len(docs)} productos en Qdrant ({collection}).")

if __name__ == "__main__":
    run()
