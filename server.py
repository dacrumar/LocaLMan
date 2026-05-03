"""
server.py
---------
Servidor FastAPI que expone el chatbot RAG como API REST con streaming.

Uso:
    python server.py
    python server.py --host 0.0.0.0 --port 8000  # accesible en red local

Endpoints:
    GET  /              → sirve el frontend (static/index.html)
    POST /chat          → respuesta completa JSON
    POST /chat/stream   → respuesta en streaming (Server-Sent Events)
    POST /reset         → reinicia la memoria conversacional
    GET  /status        → estado del servidor y modelos
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Silenciar bug de telemetría de ChromaDB/Posthog
os.environ["ANONYMIZED_TELEMETRY"] = "False"
try:
    import posthog
    posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Manuales", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ─── Estado global ────────────────────────────────────────────────────────────

chat_engine = None
ready = False
error_msg = ""

SYSTEM_PROMPT = """\
Eres un asistente técnico experto. Responde SIEMPRE basándote en el contexto \
proporcionado por los manuales. Si la información no está en los manuales, \
indícalo claramente en lugar de inventar una respuesta.
Responde en el mismo idioma en que se hace la pregunta.
"""

# ─── Modelos Pydantic ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: list[dict]

class StatusResponse(BaseModel):
    ready: bool
    error: str
    llm_model: str
    embed_model: str
    top_k: int

# ─── Inicialización ───────────────────────────────────────────────────────────

def init_engine():
    global chat_engine, ready, error_msg

    db_path = Path(config.CHROMA_PATH)
    if not db_path.exists():
        error_msg = f"No se encontró la BD vectorial en '{config.CHROMA_PATH}'. Ejecuta primero: python ingest.py"
        print(f"ERROR: {error_msg}")
        return

    try:
        embed_model = OllamaEmbedding(
            model_name=config.EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
        llm = Ollama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            request_timeout=120.0,
            system_prompt=SYSTEM_PROMPT,
        )
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        collection = client.get_collection(config.COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )
        memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            llm=llm,
            memory=memory,
            similarity_top_k=config.TOP_K,
            verbose=False,
        )
        ready = True
        print(f"✅ Motor RAG listo  (LLM: {config.LLM_MODEL} | Embeddings: {config.EMBED_MODEL})")
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR al inicializar: {e}")


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, init_engine)

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    index_file = static_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend no encontrado en static/index.html")
    return FileResponse(str(index_file))


@app.get("/status", response_model=StatusResponse)
async def status():
    return StatusResponse(
        ready=ready,
        error=error_msg,
        llm_model=config.LLM_MODEL,
        embed_model=config.EMBED_MODEL,
        top_k=config.TOP_K,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not ready:
        raise HTTPException(status_code=503, detail=error_msg or "Servidor no listo")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Mensaje vacío")

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, chat_engine.chat, req.message)

        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                meta = node.metadata or {}
                sources.append({
                    "file": meta.get("file_name", "—"),
                    "page": str(meta.get("page_label", meta.get("page", "—"))),
                    "score": round(node.score, 3) if node.score else None,
                    "snippet": node.text[:200].replace("\n", " "),
                })

        return ChatResponse(response=str(response), sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not ready:
        raise HTTPException(status_code=503, detail=error_msg or "Servidor no listo")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Mensaje vacío")

    async def event_generator():
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, chat_engine.chat, req.message)
            text = str(response)

            # Simula streaming por palabras (LlamaIndex sync no tiene stream real con chat_engine)
            words = text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.02)

            # Fuentes al final
            if hasattr(response, "source_nodes") and response.source_nodes:
                import json
                sources = []
                for node in response.source_nodes:
                    meta = node.metadata or {}
                    sources.append({
                        "file": meta.get("file_name", "—"),
                        "page": str(meta.get("page_label", meta.get("page", "—"))),
                        "score": round(node.score, 3) if node.score else None,
                    })
                yield f"data: [SOURCES]{json.dumps(sources)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR]{str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/reset")
async def reset():
    if not ready:
        raise HTTPException(status_code=503, detail="Servidor no listo")
    chat_engine.reset()
    return {"ok": True, "message": "Conversación reiniciada"}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servidor RAG Manuales")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Puerto (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload al cambiar código")
    args = parser.parse_args()

    print(f"\n🚀 Servidor arrancando en http://{args.host}:{args.port}\n")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
