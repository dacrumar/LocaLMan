"""
chat.py
-------
Chatbot interactivo que responde preguntas usando los manuales ingestados.

Uso:
    python chat.py             # modo conversación normal
    python chat.py --verbose   # muestra los chunks usados en cada respuesta
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# Silenciar telemetría de ChromaDB/Posthog
try:
    import posthog
    posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

import argparse
import sys
from pathlib import Path

import chromadb
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt
from rich import box

from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

console = Console()

SYSTEM_PROMPT = """\
Eres un asistente técnico experto. Responde SIEMPRE basándote en el contexto \
proporcionado por los manuales. Si la información no está en los manuales, \
indícalo claramente en lugar de inventar una respuesta.
Responde en el mismo idioma en que se hace la pregunta.
"""


def check_db_exists():
    db_path = Path(config.CHROMA_PATH)
    if not db_path.exists():
        console.print("[red]✗ No se encontró la base de datos vectorial.[/red]")
        console.print("  Ejecuta primero: [bold]python ingest_PyMuPDF.py[/bold]")
        sys.exit(1)


def build_chat_engine(verbose: bool):
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
    return chat_engine


def print_sources(source_nodes):
    if not source_nodes:
        return
    table = Table(
        title="Fuentes consultadas",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Archivo", style="dim", max_width=40)
    table.add_column("Pág/Hoja", justify="center", style="dim")
    table.add_column("Relevancia", justify="right")
    table.add_column("Fragmento", max_width=60, style="dim")

    for node in source_nodes:
        meta     = node.metadata or {}
        filename = meta.get("file_name", "—")
        page     = str(meta.get("page_label", meta.get("sheet_name", meta.get("page", "—"))))
        score    = f"{node.score:.3f}" if node.score else "—"
        snippet  = node.text[:120].replace("\n", " ") + "…"
        table.add_row(filename, page, score, snippet)

    console.print(table)


def chat_loop(verbose: bool):
    console.print(Panel.fit(
        "[bold cyan]RAG Manuales — Chatbot[/bold cyan]\n"
        f"[dim]Modelo: {config.LLM_MODEL}  |  Embeddings: {config.EMBED_MODEL}  |  Top-K: {config.TOP_K}[/dim]",
        border_style="cyan",
    ))
    console.print("[dim]Escribe tu pregunta y pulsa Enter. Comandos: /salir · /limpiar · /ayuda[/dim]\n")

    check_db_exists()

    console.print("[cyan]⚙  Cargando índice...[/cyan]", end=" ")
    try:
        chat_engine = build_chat_engine(verbose)
        console.print("[green]listo.[/green]\n")
    except Exception as e:
        console.print(f"\n[red]✗ Error al cargar el índice: {e}[/red]")
        console.print("  Verifica que Ollama está corriendo y que los modelos están disponibles:")
        console.print(f"  [bold]ollama pull {config.LLM_MODEL}[/bold]")
        console.print(f"  [bold]ollama pull {config.EMBED_MODEL}[/bold]")
        sys.exit(1)

    turn = 0
    while True:
        try:
            user_input = Prompt.ask("[bold green]Tú[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Hasta luego.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/salir", "/exit", "/quit", "exit", "quit"):
            console.print("[dim]Hasta luego.[/dim]")
            break

        if user_input.lower() in ("/limpiar", "/clear", "/reset"):
            chat_engine.reset()
            console.clear()
            turn = 0
            console.print("[yellow]Conversación reiniciada.[/yellow]\n")
            continue

        if user_input.lower() in ("/ayuda", "/help"):
            console.print(
                "[dim]/salir[/dim]   Terminar el chat\n"
                "[dim]/limpiar[/dim] Reiniciar la conversación\n"
                "[dim]/ayuda[/dim]   Mostrar esta ayuda\n"
            )
            continue

        turn += 1
        console.print(f"\n[bold cyan]Asistente[/bold cyan] [dim](turno {turn})[/dim]")
        try:
            response = chat_engine.chat(user_input)
        except Exception as e:
            console.print(f"[red]✗ Error al generar respuesta: {e}[/red]\n")
            continue

        console.print(Markdown(str(response)))
        console.print()

        if verbose and hasattr(response, "source_nodes") and response.source_nodes:
            print_sources(response.source_nodes)
            console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot RAG sobre manuales")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar los chunks usados en cada respuesta")
    args = parser.parse_args()
    chat_loop(verbose=args.verbose)
