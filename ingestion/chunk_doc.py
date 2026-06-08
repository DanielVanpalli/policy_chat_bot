"""
Stage 2: parsed.json -> layout-aware chunks.

Walks Docling's text blocks + tables, filters furniture (page
headers/footers, running document headers), infers section hierarchy
from numbering prefixes (I. / A. / 1. / (a)), and emits chunks that
respect the document's structure:

  * Each table stands alone as its own chunk.
  * Consecutive list items are grouped into one chunk under the active
    section heading.
  * Long paragraphs split at sentence boundaries, but never across a
    section header.
  * Every chunk is prefixed with its section breadcrumb so the
    embedding captures context.

Usage:
    python -m ingestion.chunk_doc
    python -m ingestion.chunk_doc --in dataset/parsed.json --out dataset/chunks.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Fixed paths — scripts resolve relative to themselves so `cd` doesn't matter.
_PKG_DIR = Path(__file__).resolve().parent        # .../policy_bot_rag/ingestion
DATA_DIR = _PKG_DIR / "dataset"
PARSED_JSON = DATA_DIR / "parsed.json"
CHUNKS_JSON = DATA_DIR / "chunks.json"


MAX_CHARS = 1500            # soft cap per chunk
MIN_CHARS = 80              # drop noise chunks shorter than this (tables exempt)
RUNNING_HEADER_MIN_PAGES = 3   # text verbatim on >= N pages is treated as furniture


# Docling's `level` field is unreliable on policy PDFs (headings that
# are logically nested often come back as level 1). We override whenever
# a numbering prefix is present.
_NUMBERING_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"^[IVXLCDM]+\.\s"), 1),           # I. II. III.
    (re.compile(r"^\d+\.\d+(\.\d+)?\s"), 3),       # 3.2   or 3.2.1
    (re.compile(r"^[A-Z]\.\s"), 2),                # A. B. C.
    (re.compile(r"^\d+\.\s"), 3),                  # 1. 2. 3.
    (re.compile(r"^\([a-z]\)\s"), 4),              # (a) (b)
    (re.compile(r"^\([ivx]+\)\s"), 4),             # (i) (ii)
]


def _infer_level(text: str, docling_level: int | None) -> int:
    for pattern, lvl in _NUMBERING_PATTERNS:
        if pattern.match(text):
            return lvl
    if text.isupper() and len(text) < 80:
        return 1
    return docling_level or 1


def _find_running_headers(texts: list[dict]) -> set[str]:
    """Any text block that appears verbatim on many pages is document furniture."""
    per_page: dict[str, set[int]] = {}
    for t in texts:
        body = (t.get("text") or "").strip()
        if not body or len(body) > 200:
            continue
        prov = t.get("prov") or [{}]
        page = prov[0].get("page_no")
        if page is None:
            continue
        per_page.setdefault(body, set()).add(page)
    return {t for t, pages in per_page.items() if len(pages) >= RUNNING_HEADER_MIN_PAGES}


def _table_to_markdown(tbl: dict) -> str:
    data = tbl.get("data", {})
    grid = data.get("grid") or []
    if not grid:
        return ""
    lines: list[str] = []
    for r, row in enumerate(grid):
        cells = [(c.get("text") or "").replace("\n", " ").strip() for c in row]
        lines.append("| " + " | ".join(cells) + " |")
        if r == 0:
            lines.append("|" + "|".join(["---"] * len(cells)) + "|")
    return "\n".join(lines)


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split on sentence boundaries, packing up to max_chars per part."""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf += " " + s
        else:
            parts.append(buf)
            buf = s
    if buf:
        parts.append(buf)
    return parts


def _ordered_blocks(doc: dict) -> list[dict]:
    """Merge texts + tables into a single stream, sorted by reading order."""
    blocks: list[dict] = []
    for t in doc.get("texts", []):
        prov = (t.get("prov") or [{}])[0]
        blocks.append({
            "kind": "text",
            "label": t.get("label", "text"),
            "text": t.get("text") or "",
            "level": t.get("level"),
            "page": prov.get("page_no", 0),
            "charspan": (prov.get("charspan") or [0])[0],
        })
    for i, tbl in enumerate(doc.get("tables", [])):
        prov = (tbl.get("prov") or [{}])[0]
        blocks.append({
            "kind": "table",
            "index": i,
            "markdown": _table_to_markdown(tbl),
            "page": prov.get("page_no", 0),
            "charspan": 10**9,   # place tables after in-flow text on same page
        })
    blocks.sort(key=lambda b: (b["page"], b.get("charspan", 0)))
    return blocks


def chunk(in_path: Path, out_path: Path) -> None:
    doc = json.loads(in_path.read_text(encoding="utf-8"))
    source = (doc.get("origin") or {}).get("filename", in_path.stem)
    texts = doc.get("texts", [])
    running_headers = _find_running_headers(texts)

    blocks = _ordered_blocks(doc)
    section_path: list[str] = []
    current: dict | None = None
    chunks: list[dict] = []

    def emit_current() -> None:
        nonlocal current
        if not current:
            return
        body = current["body"].strip()
        if not body:
            current = None
            return
        # Tables bypass the min-length filter — they can be small and still
        # carry essential data.
        if current["kind"] != "table" and len(body) < MIN_CHARS:
            current = None
            return

        breadcrumb = " > ".join(section_path) if section_path else ""
        prefix = f"Section: {breadcrumb}\n\n" if breadcrumb else ""
        heading = section_path[-1] if section_path else ""
        parts = [body] if current["kind"] == "table" else _split_long(body, MAX_CHARS)
        for part in parts:
            chunks.append({
                "chunk_id": f"chunk_{len(chunks):04d}",
                "content": prefix + part,
                "metadata": {
                    "chunk_type": current["kind"],
                    "section_path": list(section_path),
                    "heading": heading,
                    "page": current["page"],
                    "source": source,
                },
            })
        current = None

    def start(kind: str, page: int) -> None:
        nonlocal current
        current = {"kind": kind, "body": "", "page": page}

    for b in blocks:
        if b["kind"] == "table":
            emit_current()
            md = b["markdown"].strip()
            if not md:
                continue
            start("table", b["page"])
            current["body"] = md
            emit_current()
            continue

        label = b["label"]
        text = b["text"].strip()

        if label in {"page_header", "page_footer", "footnote"}:
            continue
        if not text or text in running_headers:
            continue
        if re.fullmatch(r"\d{1,3}", text):   # bare page number
            continue

        if label in {"section_header", "title"}:
            emit_current()
            level = _infer_level(text, b.get("level"))
            while len(section_path) >= level:
                section_path.pop()
            section_path.append(text)
            # No content yet — section intro chunks get created lazily when
            # body content arrives.
            continue

        if label == "list_item":
            if current is None or current["kind"] != "list":
                emit_current()
                start("list", b["page"])
            bullet = text if text.startswith(("-", "*", "•")) else f"- {text}"
            current["body"] += bullet + "\n"
        else:
            # paragraph / plain text
            if current is None:
                start("paragraph", b["page"])
            elif current["kind"] == "list":
                emit_current()
                start("paragraph", b["page"])
            current["body"] += text + "\n"

        if current and len(current["body"]) >= int(MAX_CHARS * 1.5):
            emit_current()

    emit_current()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks -> {out_path}")
    _print_summary(chunks, running_headers)


def _print_summary(chunks: list[dict], running_headers: set[str]) -> None:
    types = Counter(c["metadata"]["chunk_type"] for c in chunks)
    total_chars = sum(len(c["content"]) for c in chunks)
    avg_chars = total_chars // max(len(chunks), 1)
    depths = Counter(len(c["metadata"]["section_path"]) for c in chunks)

    print("\n-- chunk summary --")
    print(f"  total chunks:          {len(chunks)}")
    print(f"  avg chars/chunk:       {avg_chars}")
    print(f"  running headers dropped: {len(running_headers)}")
    print("  by type:")
    for t, n in types.most_common():
        print(f"    {t:<12} {n}")
    print("  by section depth:")
    for d in sorted(depths):
        print(f"    depth {d}: {depths[d]}")


def main() -> None:
    argparse.ArgumentParser(description="Stage 2 — turn parsed.json into layout-aware chunks.").parse_args()
    chunk(PARSED_JSON, CHUNKS_JSON)


if __name__ == "__main__":
    main()
