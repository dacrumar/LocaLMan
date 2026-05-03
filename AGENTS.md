# AGENTS.md — Arquitectura y contexto del proyecto

## Descripción general

Sistema RAG (Retrieval-Augmented Generation) local para consultar manuales técnicos
mediante un chatbot. Todo corre en local: sin APIs externas, sin enviar datos a ningún
servidor. El LLM y el modelo de embeddings se ejecutan a través de Ollama.

```
Usuario → pregunta
           ↓
       embed_model vectoriza la pregunta
           ↓
       ChromaDB busca los TOP_K chunks más cercanos (cosine similarity)
           ↓
       Se construye el prompt: [contexto chunks] + [pregunta]
           ↓
       LLM (Ollama) genera la respuesta
           ↓
       Respuesta + fuentes → usuario
```

---

## Stack tecnológico

| Capa            | Tecnología              | Rol                                              |
|-----------------|-------------------------|--------------------------------------------------|
| LLM             | Ollama + gemma3:4b      | Generación de respuestas en lenguaje natural     |
| Embeddings      | mxbai-embed-large       | Vectorización de texto (docs y queries)          |
| Vector store    | ChromaDB (persistente)  | Almacenamiento y búsqueda por similitud          |
| Orquestación    | LlamaIndex              | Pipeline RAG, chunking, chat engine con memoria  |
| Parsers PDF     | PyMuPDF (fitz)          | Extracción de texto de PDFs con layout complejo  |
| Parser DOCX     | python-docx             | Extracción de párrafos de documentos Word        |
| Parser HTML     | BeautifulSoup4          | Limpieza de HTML, eliminación de tags            |
| Parser XLSX     | openpyxl                | Conversión de filas Excel a texto narrativo      |
| Web server      | FastAPI + uvicorn       | API REST con streaming SSE para el frontend      |
| Frontend        | HTML/CSS/JS vanilla     | Chatbox web sin frameworks                       |
| Terminal UI     | Rich                    | Interfaz CLI con colores, tablas, progress bar   |

---

## Módulos

### `config.py`
Configuración central del proyecto. **Todos los parámetros ajustables están aquí.**

| Parámetro        | Default             | Descripción                                          |
|------------------|---------------------|------------------------------------------------------|
| `OLLAMA_BASE_URL` | `localhost:11434`  | URL del servidor Ollama                              |
| `LLM_MODEL`      | `gemma3:4b`         | Modelo Ollama para generar respuestas                |
| `EMBED_MODEL`    | `mxbai-embed-large` | Modelo de embeddings. **Debe ser el mismo siempre.** |
| `DOCS_DIR`       | `./manuales`        | Carpeta con los documentos fuente                    |
| `CHROMA_PATH`    | `./chroma_db`       | Carpeta donde persiste la BD vectorial               |
| `COLLECTION_NAME`| `manuales`          | Nombre de la colección en ChromaDB                   |
| `CHUNK_SIZE`     | `450`               | Tokens por chunk. Limitado por la ventana de mxbai (512 max) |
| `CHUNK_OVERLAP`  | `50`                | Solapamiento entre chunks para no perder contexto    |
| `TOP_K`          | `6`                 | Chunks recuperados por consulta                      |

> ⚠️ Si cambias `EMBED_MODEL`, debes reingestar desde cero con `--reset`.
> Los vectores de distintos modelos son incompatibles.

---

### `ingest_PyMuPDF.py`
Pipeline de ingestión multi-formato. Se ejecuta **una sola vez** (o al añadir documentos nuevos).

**Formatos soportados y parsers:**

| Extensión      | Parser       | Estrategia                                              |
|----------------|--------------|---------------------------------------------------------|
| `.pdf`         | PyMuPDF      | Una página = un Document. Filtra líneas < 3 chars       |
| `.docx`        | python-docx  | Extrae párrafos no vacíos como un único Document        |
| `.html`/`.htm` | BeautifulSoup| Elimina script/style/nav, extrae texto plano            |
| `.xlsx`        | openpyxl     | Cada fila → "Col1: val1 \| Col2: val2" para preservar contexto |
| `.txt`/`.md`   | built-in     | Lectura directa UTF-8                                   |

**Flags:**
- `--reset`: borra la colección de ChromaDB antes de ingestar (necesario al cambiar modelo de embeddings)
- `--debug`: imprime los primeros 3 chunks para inspeccionar que el parseo es correcto

**Flujo interno:**
1. Detecta archivos en `DOCS_DIR` por extensión
2. Aplica el parser correspondiente → lista de `Document` con metadatos
3. `SentenceSplitter` trocea en chunks de `CHUNK_SIZE` tokens con `CHUNK_OVERLAP`
4. `OllamaEmbedding` vectoriza cada chunk en lotes de 20
5. Los vectores se almacenan en ChromaDB con sus metadatos

---

### `chat.py`
Chatbot interactivo en terminal.

- Carga el índice desde ChromaDB (no re-ingestar)
- Usa `ChatMemoryBuffer` para mantener contexto conversacional (últimos 4096 tokens)
- `condense_plus_context`: condensa el historial de la conversación para refinar la query antes de buscar en ChromaDB
- `--verbose`: muestra tabla de fuentes (archivo, página, score de relevancia, fragmento) tras cada respuesta
- Comandos: `/salir`, `/limpiar`, `/ayuda`

---

### `server.py`
Backend FastAPI que expone el chatbot como servicio web.

**Endpoints:**

| Método | Ruta           | Descripción                                    |
|--------|----------------|------------------------------------------------|
| GET    | `/`            | Sirve el frontend (static/index.html)          |
| GET    | `/status`      | Estado del servidor, modelos y configuración   |
| POST   | `/chat`        | Respuesta completa en JSON                     |
| POST   | `/chat/stream` | Respuesta en streaming (Server-Sent Events)    |
| POST   | `/reset`       | Reinicia la memoria conversacional             |

El streaming funciona por palabras con un delay de 20ms entre tokens simulados,
ya que LlamaIndex sync no expone stream real en `chat_engine.chat()`.

**Arranque:**
```bash
python server.py                          # localhost:8000
python server.py --host 0.0.0.0 --port 8080  # accesible en red local
```

---

### `static/index.html`
Frontend vanilla (HTML + CSS + JS, sin frameworks).

- Streaming word-by-word vía EventSource / fetch + ReadableStream
- Fuentes colapsables por respuesta (archivo, página, score)
- Memoria conversacional (botón "nueva conversación" llama a `/reset`)
- Pill de estado en header (conectando / modelo activo / error)
- Auto-resize del textarea
- Enter para enviar, Shift+Enter para nueva línea

---

## Notas importantes

### Por qué mxbai-embed-large y no nomic-embed-text
`nomic-embed-text` es un modelo general que produce scores bajos en documentación
técnica con terminología específica (SAP, infra, networking...). `mxbai-embed-large`
rinde mejor en ese dominio pero tiene una ventana de contexto máxima de **512 tokens**,
por eso `CHUNK_SIZE` debe mantenerse en 450 o menos.

### El modelo de embeddings debe ser siempre el mismo
Los vectores en ChromaDB son generados por un modelo concreto. Si cambias `EMBED_MODEL`
en `config.py`, los vectores existentes son incompatibles con las nuevas queries.
Siempre ejecuta `ingest_PyMuPDF.py --reset` al cambiar de modelo.

### Excel (XLSX): limitaciones
El contenido tabular no se vectoriza bien si las celdas se tratan como texto aislado.
El parser convierte cada fila en una línea narrativa `"Col: val | Col: val"` para darle
contexto semántico, pero si el Excel tiene estructura muy compleja o datos numéricos
sin cabeceras descriptivas, los resultados serán pobres. En ese caso es mejor exportar
a CSV/TXT con descripciones manuales antes de ingestar.

---

## Flujo de trabajo habitual

```bash
# Primera vez
pip install -r requirements.txt
ollama pull mxbai-embed-large
ollama pull gemma3:4b

# Colocar documentos en ./manuales/ y ingestar
python ingest_PyMuPDF.py

# Opción A: chatbot en terminal
python chat.py --verbose

# Opción B: interfaz web
python server.py
# → abrir http://localhost:8000

# Al añadir documentos nuevos
python ingest_PyMuPDF.py          # añade sin borrar lo anterior

# Al cambiar modelo de embeddings
python ingest_PyMuPDF.py --reset  # borra BD y reinicia desde cero
```
