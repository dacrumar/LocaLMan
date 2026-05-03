# RAG Manuales

Chatbot local que responde preguntas sobre tus manuales usando RAG.
Todo corre en local: sin APIs externas, sin enviar datos a ningún servidor.

```
Pregunta → Embeddings → ChromaDB → Top-K chunks → Ollama/LLM → Respuesta
```

## Requisitos

- Python 3.10+
- [Ollama](https://ollama.com) corriendo en local

## Instalación

```bash
pip install -r requirements.txt
```

Descarga los modelos necesarios en Ollama:

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull gemma3:4b          # LLM (o el que prefieras)
```

## Uso

### 1. Coloca los manuales

Pon tus archivos PDF, DOCX, TXT o Markdown en la carpeta `./manuales/`.

### 2. Ingestión (una sola vez)

```bash
python ingest.py
```

Si quieres borrar la BD y empezar de cero:

```bash
python ingest.py --reset
```

### 3. Chatbot

```bash
python chat.py              # modo normal
python chat.py --verbose    # muestra qué fragmentos usó en cada respuesta
```

### Comandos dentro del chat

| Comando    | Acción                        |
|------------|-------------------------------|
| `/salir`   | Terminar el chat              |
| `/limpiar` | Reiniciar la conversación     |
| `/ayuda`   | Mostrar ayuda                 |

## Configuración

Edita `config.py` para cambiar:

| Parámetro        | Descripción                                  | Default          |
|------------------|----------------------------------------------|------------------|
| `LLM_MODEL`      | Modelo Ollama para respuestas                | `gemma3:4b`      |
| `EMBED_MODEL`    | Modelo Ollama para embeddings                | `nomic-embed-text` |
| `CHUNK_SIZE`     | Tokens por chunk al trocear documentos       | `512`            |
| `CHUNK_OVERLAP`  | Solapamiento entre chunks consecutivos       | `50`             |
| `TOP_K`          | Chunks recuperados por consulta              | `4`              |
| `DOCS_DIR`       | Carpeta de manuales                          | `./manuales`     |
| `CHROMA_PATH`    | Carpeta de la BD vectorial                   | `./chroma_db`    |

## Estructura del proyecto

```
rag-manuales/
├── manuales/       ← pon aquí tus PDFs/DOCX/TXT
├── chroma_db/      ← se genera automáticamente al ingestar
├── config.py       ← configuración central
├── ingest.py       ← pipeline de ingestión
├── chat.py         ← chatbot interactivo
└── requirements.txt
```

## Añadir nuevos manuales

Copia los nuevos archivos a `./manuales/` y vuelve a ejecutar `ingest.py`.
Los documentos ya existentes no se re-vectorizan (ChromaDB los ignora por ID).
