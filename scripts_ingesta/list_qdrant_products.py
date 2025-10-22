import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = "catalog_kb"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

print(f"Consultando productos en Qdrant ({COLLECTION})...")

# Recupera los primeros 10 puntos/productos
response = client.scroll(
    collection_name=COLLECTION,
    limit=10,
    with_payload=True,
)

points = response[0]
if not points:
    print("No se encontraron productos en Qdrant.")
else:
    for i, point in enumerate(points, 1):
        print(f"\nProducto {i}:")
        payload = point.payload or {}
        for k, v in payload.items():
            print(f"  {k}: {v}")
