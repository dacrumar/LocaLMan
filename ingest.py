"""
ingest.py
-----------------
Versión mejorada del pipeline de ingestión usando PyMuPDF como parser de PDFs.
Más robusto para documentos con layouts complejos, columnas, tablas o texto escaneado.

Uso:
    python ingest.py                  # procesa todo ./manuales
    python ingest.py --reset          # borra la BD y reinicia desde cero
    python ingest.py --debug          # imprime los primeros 3 chunks de cada PDF

Requiere:
    pip install pymupdf llama-index-readers-file
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# Silenciar telemetría de ChromaDB/Posthog
try:
    import posthog
    posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass
import sys
import argparse
from pathlib import Path

import chromadb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

console = Console()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_embed_model():
    return OllamaEmbedding(
        model_name=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def get_chroma_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            console.print("[yellow]⚠  Colección anterior eliminada.[/yellow]")
        except Exception:
            pass
    collection = client.get_or_create_collection(config.COLLECTION_NAME)
    return client, collection


# ─── Carga de documentos con PyMuPDF ──────────────────────────────────────────

def load_documents_pymupdf() -> list[Document]:
    """
    Carga PDFs usando PyMuPDF (fitz) directamente.
    Cada página se convierte en un Document independiente con metadatos enriquecidos.
    También carga TXT y MD con el reader estándar.
    """
    docs_path = Path(config.DOCS_DIR)
    if not docs_path.exists() or not any(docs_path.iterdir()):
        console.print(f"[red]✗ No hay documentos en '{config.DOCS_DIR}'.[/red]")
        console.print("  Coloca PDFs, TXT o MD en esa carpeta y vuelve a ejecutar.")
        sys.exit(1)

    try:
        import fitz  # PyMuPDF
    except ImportError:
        console.print("[red]✗ PyMuPDF no está instalado.[/red]")
        console.print("  Ejecuta: [bold]pip install pymupdf[/bold]")
        sys.exit(1)

    documents: list[Document] = []
    pdf_files  = sorted(docs_path.rglob("*.pdf"))
    text_files = sorted(docs_path.rglob("*.txt")) + sorted(docs_path.rglob("*.md"))

    if not pdf_files and not text_files:
        console.print("[red]✗ No se encontraron archivos PDF, TXT ni MD.[/red]")
        sys.exit(1)

    # ── PDFs con PyMuPDF ──
    if pdf_files:
        console.print(f"[cyan]📄 Parseando {len(pdf_files)} PDF(s) con PyMuPDF...[/cyan]")

    for pdf_path in pdf_files:
        try:
            doc = fitz.open(str(pdf_path))
            pages_loaded = 0
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extrae texto con preservación de layout (mejor para columnas)
                text = page.get_text("text")

                # Limpieza básica: elimina líneas muy cortas (cabeceras/pies de página)
                lines = text.split("\n")
                lines = [l for l in lines if len(l.strip()) > 3]
                text = "\n".join(lines).strip()

                if not text:
                    continue  # página vacía o solo imágenes

                documents.append(Document(
                    text=text,
                    metadata={
                        "file_name":  pdf_path.name,
                        "file_path":  str(pdf_path),
                        "page_label": str(page_num + 1),
                        "page":       page_num + 1,
                        "total_pages": len(doc),
                        "source":     "pymupdf",
                    }
                ))
                pages_loaded += 1

            doc.close()
            console.print(f"  [green]✓[/green] {pdf_path.name} — {pages_loaded} páginas")

        except Exception as e:
            console.print(f"  [red]✗[/red] {pdf_path.name} — Error: {e}")

    # ── TXT / MD con reader estándar ──
    if text_files:
        console.print(f"\n[cyan]📝 Cargando {len(text_files)} archivo(s) de texto...[/cyan]")
        from llama_index.core import SimpleDirectoryReader
        for tf in text_files:
            try:
                reader = SimpleDirectoryReader(input_files=[str(tf)])
                docs = reader.load_data()
                documents.extend(docs)
                console.print(f"  [green]✓[/green] {tf.name}")
            except Exception as e:
                console.print(f"  [red]✗[/red] {tf.name} — Error: {e}")

    console.print()
    return documents


# ─── Split ────────────────────────────────────────────────────────────────────

def split_documents(documents: list[Document]):
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=False)
    return nodes


# ─── Debug ────────────────────────────────────────────────────────────────────

def print_debug(nodes, n: int = 3):
    console.print(Rule("[yellow]DEBUG — primeros chunks[/yellow]"))
    for i, node in enumerate(nodes[:n]):
        meta = node.metadata or {}
        console.print(f"\n[bold yellow]Chunk {i}[/bold yellow]  "
                       f"[dim]{meta.get('file_name','?')} · pág. {meta.get('page_label','?')}[/dim]")
        console.print(node.text[:600])
        console.print()
    console.print(Rule())


# ─── Resumen ──────────────────────────────────────────────────────────────────

def print_summary(documents, nodes):
    table = Table(title="Resumen de ingestión", show_header=True, header_style="bold cyan")
    table.add_column("Métrica", style="dim")
    table.add_column("Valor", justify="right")

    files = {doc.metadata.get("file_name", "desconocido") for doc in documents}
    table.add_row("Páginas / documentos leídos", str(len(documents)))
    table.add_row("Archivos únicos",             str(len(files)))
    table.add_row("Chunks generados",            str(len(nodes)))
    table.add_row("Chunk size",                  str(config.CHUNK_SIZE))
    table.add_row("Chunk overlap",               str(config.CHUNK_OVERLAP))
    table.add_row("Parser PDF",                  "PyMuPDF")
    table.add_row("Modelo embeddings",           config.EMBED_MODEL)
    table.add_row("LLM",                         config.LLM_MODEL)

    console.print(table)
    console.print()
    console.print("[bold]Archivos procesados:[/bold]")
    for f in sorted(files):
        console.print(f"  [green]✓[/green] {f}")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def ingest(reset: bool = False, debug: bool = False):
    console.print(Panel.fit(
        "[bold cyan]RAG Manuales — Ingestión (PyMuPDF)[/bold cyan]",
        border_style="cyan"
    ))

    # 1. Verificar Ollama
    console.print("[cyan]🔌 Verificando conexión con Ollama...[/cyan]")
    try:
        embed_model = get_embed_model()
        embed_model.get_text_embedding("test")
        console.print(f"  [green]✓[/green] Ollama OK  ({config.EMBED_MODEL})\n")
    except Exception as e:
        console.print(f"[red]✗ No se puede conectar con Ollama: {e}[/red]")
        console.print("  Asegúrate de que Ollama está corriendo y de haber ejecutado:")
        console.print(f"  [bold]ollama pull {config.EMBED_MODEL}[/bold]")
        sys.exit(1)

    # 2. Cargar con PyMuPDF
    documents = load_documents_pymupdf()

    # 3. Trocear
    console.print(f"[cyan]✂  Troceando en chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})...[/cyan]")
    nodes = split_documents(documents)
    console.print(f"  [green]✓[/green] {len(nodes)} chunks listos\n")

    # 4. Debug opcional
    if debug:
        print_debug(nodes)

    # 5. ChromaDB
    _, collection = get_chroma_collection(reset=reset)
    vector_store  = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 6. Embeddings y guardado
    console.print("[cyan]🧮 Generando embeddings y almacenando en ChromaDB...[/cyan]")
    console.print("  (puede tardar varios minutos según el volumen)\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Vectorizando chunks...", total=len(nodes))
        batch_size = 20
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            VectorStoreIndex(
                batch,
                storage_context=storage_context,
                embed_model=embed_model,
            )
            progress.update(task, advance=len(batch))

    console.print()
    console.print("[bold green]✅ Ingestión completada.[/bold green]")
    console.print(f"   Base de datos guardada en: [bold]{config.CHROMA_PATH}[/bold]\n")

    print_summary(documents, nodes)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestar manuales en ChromaDB usando PyMuPDF")
    parser.add_argument("--reset", action="store_true",
                        help="Borrar la BD existente antes de ingestar")
    parser.add_argument("--debug", action="store_true",
                        help="Imprimir los primeros 3 chunks para inspeccionar el parseo")
    args = parser.parse_args()
    ingest(reset=args.reset, debug=args.debug)