# ─────────────────────────────────────────
#  Configuración central del proyecto
# ─────────────────────────────────────────

OLLAMA_BASE_URL   = "http://localhost:11434"

# Modelo LLM para responder preguntas
LLM_MODEL         = "gemma4-opencode:latest"          # cámbialo si usas otro

# Modelo de embeddings (solo para vectorizar texto)
EMBED_MODEL       = "mxbai-embed-large"   # ollama pull mxbai-embed-large

# Carpeta donde están los manuales
DOCS_DIR          = "./manuales"

# Carpeta donde se persiste la base de datos vectorial
CHROMA_PATH       = "./chroma_db"

# Nombre de la colección dentro de ChromaDB
COLLECTION_NAME   = "manuales"

# Tamaño de cada chunk en tokens
CHUNK_SIZE        = 450

# Solapamiento entre chunks consecutivos
CHUNK_OVERLAP     = 100

# Cuántos chunks recuperar por consulta (más = más contexto, más lento)
TOP_K             = 6
