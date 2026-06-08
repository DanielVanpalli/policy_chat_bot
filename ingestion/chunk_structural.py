"""
Stage 2 (alternative): parsed.json -> structure-aware chunks -> FAISS HNSW index.

Walks the Docling tree in reading order, starts a new chunk at every level-1
or level-2 section header, and inlines list groups. Each chunk carries the
section path and page range as metadata so the retriever can surface context
that already knows where it came from.

Vector index is FAISS IndexHNSWFlat (graph-based ANN). Faster than flat L2
once the corpus grows past a few thousand chunks; for this repo it's future-
proofing. Tune HNSW_* constants below if recall or latency shifts.

Outputs (kept separate from build_index.py so both can coexist):
  - dataset/chunks_structural/   one .txt per chunk for human inspection
  - dataset/chunks_structural/chunks.jsonl   text + metadata, canonical form
  - faiss_index_structural/   FAISS HNSW index with metadata

Usage:
    python -m ingestion.chunk_structural
    python -m ingestion.chunk_structural --parsed dataset/parsed.json \\
        --out faiss_index_structural --chunks-dir dataset/chunks_structural
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


EMBED_MODEL = "all-MiniLM-L6-v2"
SECTION_BREAK_LEVEL = 2          # start a new chunk on headers with level <= this
MAX_CHARS = 3200                 # ~800 tokens — sub-split sections larger than this
MIN_CHARS = 200                  # merge a trailing runt into the previous chunk
SKIP_LABELS = {"page_footer", "picture", "page_header", "unspecified"}

# HNSW tuning — M is graph connectivity, efConstruction is build-time search
# depth (accuracy), efSearch is query-time depth (recall/latency tradeoff).
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


# ── tree resolution ──────────────────────────────────────────────────────────

def _resolver(doc: dict) -> callable:
    """Return a function that turns a JSON-pointer string like '#/texts/5'
    into the referenced object. parsed.json uses these refs everywhere."""
    def resolve(ref: str) -> dict:
        # ref looks like "#/texts/5" — split to ["", "texts", "5"]
        parts = ref.lstrip("#/").split("/")
        node: Any = doc
        for p in parts:
            node = node[int(p)] if p.isdigit() else node[p]
        return node
    return resolve


@dataclass
class Item:
    """One atomic piece of content in reading order."""
    text: str
    label: str
    page_no: int | None
    level: int | None = None      # only set for section_header


def _first_page(node: dict) -> int | None:
    for p in node.get("prov", []) or []:
        if "page_no" in p:
            return p["page_no"]
    return None


def flatten(doc: dict) -> list[Item]:
    """Walk body.children in reading order, expanding groups inline.

    Section headers that repeat across pages (e.g. a company letterhead) are
    demoted to plain text so they don't fragment the section hierarchy."""
    resolve = _resolver(doc)
    items: list[Item] = []

    # first pass: count how often each section_header text appears. Any header
    # that shows up more than once is almost certainly a page letterhead, not
    # a real section boundary.
    header_counts: dict[str, int] = {}
    for child in doc["body"]["children"]:
        _count_headers(resolve(child["$ref"]), resolve, header_counts)
    repeated_headers = {t for t, c in header_counts.items() if c > 1}

    def visit(node: dict) -> None:
        label = node.get("label")
        children = node.get("children", []) or []

        if label == "list" or node.get("name") == "_root_":
            for child in children:
                visit(resolve(child["$ref"]))
            return

        if label in SKIP_LABELS:
            return

        text = (node.get("text") or "").strip()
        if not text:
            return

        # demote a repeated header so it doesn't reset the section path
        effective_label = label or "text"
        effective_level = node.get("level") if label == "section_header" else None
        if label == "section_header" and text in repeated_headers:
            effective_label = "text"
            effective_level = None

        items.append(Item(
            text=text,
            label=effective_label,
            page_no=_first_page(node),
            level=effective_level,
        ))

    for child in doc["body"]["children"]:
        visit(resolve(child["$ref"]))

    return items


def _count_headers(node: dict, resolve, counts: dict[str, int]) -> None:
    if node.get("label") == "section_header":
        txt = (node.get("text") or "").strip()
        if txt:
            counts[txt] = counts.get(txt, 0) + 1
    for child in node.get("children", []) or []:
        _count_headers(resolve(child["$ref"]), resolve, counts)


# ── chunking ─────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    section_path: list[str] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)

    def render(self) -> str:
        """Plain-text form fed to the embedder and the LLM."""
        header = " > ".join(self.section_path) if self.section_path else ""
        body_parts: list[str] = []
        for it in self.items:
            if it.label == "section_header":
                body_parts.append(f"\n## {it.text}\n")
            elif it.label == "list_item":
                body_parts.append(f"- {it.text}")
            else:
                body_parts.append(it.text)
        body = "\n".join(body_parts).strip()
        return f"[Section: {header}]\n\n{body}" if header else body

    def char_len(self) -> int:
        return sum(len(it.text) for it in self.items)

    def page_range(self) -> tuple[int | None, int | None]:
        pages = [it.page_no for it in self.items if it.page_no is not None]
        return (min(pages), max(pages)) if pages else (None, None)


def chunk_items(items: list[Item]) -> list[Chunk]:
    """Start a new chunk at every level<=SECTION_BREAK_LEVEL header. Sub-split
    sections that exceed MAX_CHARS along paragraph boundaries. Merge runts.

    Each chunk's section_path is the header stack at the moment it was opened,
    so chunks keep their context even when the enclosing section ends."""
    chunks: list[Chunk] = []
    section_stack: list[tuple[int, str]] = []   # [(level, title), ...]
    current = Chunk()

    def open_chunk() -> None:
        """Close the current chunk (if non-empty) and start a new one with
        section_path snapshotted from the current stack."""
        nonlocal current
        if current.items:
            chunks.append(current)
        current = Chunk(section_path=[title for _, title in section_stack])

    for it in items:
        if it.label == "section_header" and it.level is not None:
            # prune + push happen first so the new chunk inherits the updated path
            section_stack = [(lv, t) for lv, t in section_stack if lv < it.level]
            section_stack.append((it.level, it.text))
            if it.level <= SECTION_BREAK_LEVEL:
                open_chunk()
            current.items.append(it)
            continue

        # size-based sub-split inside a long section
        if current.char_len() + len(it.text) > MAX_CHARS and current.items:
            open_chunk()

        current.items.append(it)

    if current.items:
        chunks.append(current)

    # merge a trailing runt into the previous chunk to avoid near-empty chunks
    merged: list[Chunk] = []
    for ch in chunks:
        if merged and ch.char_len() < MIN_CHARS and merged[-1].section_path == ch.section_path:
            merged[-1].items.extend(ch.items)
        else:
            merged.append(ch)
    return merged


# ── index build ──────────────────────────────────────────────────────────────

def _build_hnsw_store(docs: list[Document], embeddings) -> FAISS:
    """Build an HNSW-backed FAISS vectorstore.

    LangChain's FAISS.from_documents defaults to IndexFlatL2. For HNSW we
    construct faiss.IndexHNSWFlat ourselves and hand it to the wrapper, then
    add documents through the normal add path so the docstore stays in sync."""
    dim = len(embeddings.embed_query("dim probe"))
    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    store.add_documents(docs)
    return store


def to_documents(chunks: list[Chunk], source: str) -> list[Document]:
    docs: list[Document] = []
    for i, ch in enumerate(chunks):
        p_start, p_end = ch.page_range()
        labels = sorted({it.label for it in ch.items})
        docs.append(Document(
            page_content=ch.render(),
            metadata={
                "chunk_id": i,
                "source": source,
                "section_path": " > ".join(ch.section_path),
                "page_start": p_start,
                "page_end": p_end,
                "labels": ",".join(labels),
                "char_len": ch.char_len(),
            },
        ))
    return docs


def build(parsed_path: Path, index_dir: Path, chunks_dir: Path) -> None:
    if not parsed_path.exists():
        raise SystemExit(f"parsed.json not found: {parsed_path} - run parse_doc first")

    doc = json.loads(parsed_path.read_text(encoding="utf-8"))
    items = flatten(doc)
    print(f"Flattened {len(items)} content items from {parsed_path}")

    chunks = chunk_items(items)
    print(f"Produced {len(chunks)} structural chunks")

    chunks_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = chunks_dir / "chunks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for i, ch in enumerate(chunks):
            rendered = ch.render()
            (chunks_dir / f"chunk_{i}.txt").write_text(rendered, encoding="utf-8")
            p_start, p_end = ch.page_range()
            jf.write(json.dumps({
                "chunk_id": i,
                "section_path": ch.section_path,
                "page_start": p_start,
                "page_end": p_end,
                "char_len": ch.char_len(),
                "text": rendered,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote chunks -> {chunks_dir}/ ({jsonl_path.name} is canonical)")

    docs = to_documents(chunks, source=str(parsed_path))
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = _build_hnsw_store(docs, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"Saved FAISS HNSW index -> {index_dir}/ "
          f"(M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}, efSearch={HNSW_EF_SEARCH})")

    _summary(chunks)


def _summary(chunks: list[Chunk]) -> None:
    lens = [ch.char_len() for ch in chunks]
    print("\n-- chunk summary --")
    print(f"  count:   {len(chunks)}")
    if lens:
        print(f"  chars:   min={min(lens)}  p50={sorted(lens)[len(lens)//2]}  max={max(lens)}")
    with_section = sum(1 for ch in chunks if ch.section_path)
    print(f"  with section path: {with_section}/{len(chunks)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", type=Path, default=Path("dataset/parsed.json"))
    ap.add_argument("--out", dest="index_dir", type=Path, default=Path("faiss_index_structural"))
    ap.add_argument("--chunks-dir", type=Path, default=Path("dataset/chunks_structural"))
    args = ap.parse_args()
    build(args.parsed, args.index_dir, args.chunks_dir)


if __name__ == "__main__":
    main()
