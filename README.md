# agent_search_product

Agente de búsqueda de productos con RAG y clasificación automática por IA. Compatible con Gemini, OpenAI y Ollama.

---

## Comandos recomendados para levantar todo el stack

1. **Instala podman-compose si no lo tienes:**
   ```
   pip install podman-compose
   ```

2. **Levanta Qdrant (versión compatible) con Podman:**
   ```
   podman-compose -f agent_search_product/docker-compose-qdrant.yml up -d
   ```

3. **Crea y activa el entorno virtual de Python:**
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Instala dependencias del proyecto:**
   ```
   pip install -r requirements.txt
   ```

5. **Configura el archivo `.env` en `agent_search_product/`**  
   - Usa `EMBEDDINGS_PROVIDER=ollama` y `OLLAMA_MODEL=nomic-embed-text` para embeddings locales.
   - Usa `LLM_PROVIDER=gemini` para clasificación con Gemini.

6. **Asegúrate de tener el modelo de embeddings en Ollama:**
   ```
   ollama pull nomic-embed-text
   ```

7. **Ingesta productos en Qdrant:**
   ```
   python3 agent_search_product/scripts_ingesta/ingest_catalog.py
   ```

8. **(Opcional) Verifica los productos ingresados:**
   ```
   python3 agent_search_product/scripts_ingesta/list_qdrant_products.py
   ```

9. **Levanta el agente de productos (API FastAPI):**
   ```
   python3 agent_search_product/main.py
   ```

---

## Notas importantes

- Si ves un warning deprecado sobre `OllamaEmbeddings`, puedes ignorarlo por ahora.  
  Cuando actualices todos los paquetes langchain, cambia el import a:
  ```python
  from langchain_ollama import OllamaEmbeddings
  ```
- Qdrant debe estar corriendo en la versión 1.15.1 para ser compatible con el cliente Python.
- Si usas Podman, asegúrate de tener podman-compose instalado.
- El archivo de catálogo debe estar en `agent_search_product/data/catalog_samples.csv`.

---

## Estructura

- `main.py`: API principal del agente de productos.
- `vector/`: Lógica de búsqueda vectorial y clasificación por IA.
- `scripts_ingesta/`: Scripts para ingestar y listar productos en Qdrant.
  - `ingest_catalog.py`: Script para ingestar productos.
  - `list_qdrant_products.py`: Script para listar productos ya ingresados.
- `.env.example`: Variables de entorno de ejemplo.
- `docker-compose-qdrant.yml`: Compose para levantar Qdrant.
- `data/`: Carpeta para el archivo de catálogo CSV.

---

## Troubleshooting

- **Qdrant version mismatch:**  
  Si ves un error de incompatibilidad de versiones, asegúrate de que Qdrant esté en la versión 1.15.1 (`docker-compose-qdrant.yml` ya lo especifica).
- **Ollama model not found:**  
  Si ves un error de modelo no encontrado, ejecuta `ollama pull nomic-embed-text`.
- **DeprecationWarning de embeddings:**  
  Es solo un warning, puedes ignorarlo hasta actualizar todos los paquetes langchain.

---

Sigue estos pasos y comandos para tener todo el stack funcionando correctamente.
