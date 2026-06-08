"""
Stage 1: PDF -> structured document.

Uses Docling to parse the policy PDF into a layout-aware document tree
(headings, paragraphs, tables, lists, page numbers). No LLM involved —
Docling runs small local vision/layout models that auto-download on
first use.

Outputs:
  - dataset/parsed.json   — full structured tree, fed to stage 2 (chunker)
  - dataset/parsed.md     — markdown preview for human inspection

Usage:
    python -m ingestion.parse_doc
    python -m ingestion.parse_doc --pdf dataset/policy_dc.pdf --out-dir dataset
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Fixed paths — scripts resolve relative to themselves so `cd` doesn't matter.
_PKG_DIR = Path(__file__).resolve().parent        # .../policy_bot_rag/ingestion
DATA_DIR = _PKG_DIR / "dataset"
PDF_PATH = DATA_DIR / "policy_dc.pdf"

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def _build_converter(
    force_full_ocr: bool,
    do_ocr: bool,
    backend: str = "pypdfium",
    num_threads: int = 1,
) -> DocumentConverter:
    """Memory-conscious pipeline for policy PDFs.

    The default ``docling-parse`` backend (C++) routinely throws
    std::bad_alloc on pages with large embedded images or dense layouts.
    PyPdfiumDocumentBackend (pure Python) resolves ~90% of those cases at
    the cost of slightly weaker table-structure recovery.

    num_threads=1 further caps peak memory by running the pipeline
    sequentially. Bump it if you have RAM to spare and want speed.

    Enable --force-full-ocr only when the text layer is unreliable; pass
    --no-ocr when the PDF is fully digital to skip OCR entirely."""
    opts = PdfPipelineOptions()
    opts.do_ocr = do_ocr
    opts.do_table_structure = Tue
    opts.table_structure_options.mode = TableFormerMode.FAST
    opts.table_structure_options.do_cell_matching = True
    opts.images_scale = 1.0
    opts.accelerator_options = AcceleratorOptions(num_threads=num_threads)
    if force_full_ocr:
        opts.ocr_options.force_full_page_ocr = True

    format_opt_kwargs: dict = {"pipeline_options": opts}
    if backend == "pypdfium":
        format_opt_kwargs["backend"] = PyPdfiumDocumentBackend
    # backend="default" leaves Docling's default (docling-parse) in place

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(**format_opt_kwargs)}
    )


def _pdf_page_count(pdf_path: Path) -> int:
    """Cheap page count using pypdfium2 — Docling already depends on it."""
    import pypdfium2
    pdf = pypdfium2.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def _merge_dicts(base: dict, incoming: dict) -> dict:
    """Concatenate texts, tables, pictures, pagesr from a batch into base.

    Internal refs (``body.children``, ``groups``) are dropped — stage 2
    walks texts + tables directly using page/charspan, so we don't need
    to preserve the cross-reference graph."""
    if not base:
        base = {
            "schema_name": incoming.get("schema_name"),
            "version": incoming.get("version"),
            "name": incoming.get("name"),
            "origin": incoming.get("origin"),
            "texts": [],
            "tables": [],
            "pictures": [],
            "pages": {},
        }
    base["texts"].extend(incoming.get("texts", []))
    base["tables"].extend(incoming.get("tables", []))
    base["pictures"].extend(incoming.get("pictures", []))
    base["pages"].update(incoming.get("pages", {}))
    return base


def _parse_skip_pages(spec: str | None) -> set[int]:
    """Parse a skip-pages spec like '21-33,40,45-47' into a set of page numbers."""
    if not spec:
        return set()
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def parse(
    pdf_path: Path,
    out_dir: Path,
    force_full_ocr: bool = False,
    do_ocr: bool = True,
    batch_size: int | None = None,
    skip_pages: set[int] | None = None,
    backend: str = "pypdfium",
    num_threads: int = 1,
) -> None:
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "parsed.json"
    md_path = out_dir / "parsed.md"

    print(f"Parsing {pdf_path} ...")
    print(f"  backend={backend}  threads={num_threads}  ocr={do_ocr}  force_full_ocr={force_full_ocr}  table_mode=FAST  images_scale=1.0")

    total_pages = _pdf_page_count(pdf_path)
    skip_pages = skip_pages or set()
    pages_to_process = [p for p in range(1, total_pages + 1) if p not in skip_pages]

    if batch_size is None or batch_size >= len(pages_to_process):
        # One batch spanning all retained pages. If we're skipping pages in the
        # middle, fall back to per-page batches so skipped pages don't break the range.
        if skip_pages:
            batches = [(p, p) for p in pages_to_process]
        else:
            batches = [(1, total_pages)]
    else:
        batches = []
        i = 0
        while i < len(pages_to_process):
            chunk = pages_to_process[i : i + batch_size]
            # Only group pages if they're contiguous — Docling page_range is a tuple.
            contiguous = [chunk[0]]
            for p in chunk[1:]:
                if p == contiguous[-1] + 1:
                    contiguous.append(p)
                else:
                    break
            batches.append((contiguous[0], contiguous[-1]))
            i += len(contiguous)

    print(f"  total pages: {total_pages}  skip: {sorted(skip_pages) or '[]'}  batches: {len(batches)}")

    t0 = time.perf_counter()
    converter = _build_converter(force_full_ocr, do_ocr, backend=backend, num_threads=num_threads)

    merged: dict = {}
    md_parts: list[str] = []
    failed_batches: list[tuple[int, int]] = []

    for start, end in batches:
        print(f"  [batch {start}-{end}] ...")
        try:
            result = converter.convert(str(pdf_path), page_range=(start, end))
            doc = result.document
            merged = _merge_dicts(merged, doc.export_to_dict())
            md_parts.append(doc.export_to_markdown())
        except Exception as exc:
            print(f"    [!] batch failed: {exc}")
            failed_batches.append((start, end))

    elapsed = time.perf_counter() - t0
    print(f"Parsed in {elapsed:.1f}s")
    if failed_batches:
        print(f"  [!] failed batches: {failed_batches} — rerun with smaller --batch-size")

    json_path.write_text(
        json.dumps(merged, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote structured tree -> {json_path} ({json_path.stat().st_size:,} bytes)")

    md_path.write_text("\n\n".join(md_parts), encoding="utf-8")
    print(f"Wrote markdown preview -> {md_path}")

    _print_summary(merged)


def _print_summary(doc_dict: dict) -> None:
    """Rough counts so you can sanity-check the extraction."""
    texts = doc_dict.get("texts", [])
    tables = doc_dict.get("tables", [])
    pictures = doc_dict.get("pictures", [])
    pages = doc_dict.get("pages", {})

    label_counts: dict[str, int] = {}
    for t in texts:
        label = t.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\n-- extraction summary --")
    print(f"  pages:     {len(pages)}")
    print(f"  tables:    {len(tables)}")
    print(f"  pictures:  {len(pictures)}")
    print(f"  text blocks by label:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {label:<20} {count}")

    if tables:
        print("\n  tables by page:")
        per_page: dict[int, int] = {}
        for tbl in tables:
            for p in tbl.get("prov", []):
                pg = p.get("page_no")
                if pg is not None:
                    per_page[pg] = per_page.get(pg, 0) + 1
        for pg in sorted(per_page):
            print(f"    page {pg}: {per_page[pg]}")
    else:
        print("\n  [!] no tables detected - rerun with --force-full-ocr if PDF has rasterized tables")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 — parse the policy PDF into a layout-aware tree.")
    ap.add_argument("--force-full-ocr", action="store_true",
                    help="OCR every page end-to-end (slow). Use when tables are rasterized.")
    ap.add_argument("--no-ocr", action="store_true",
                    help="Disable OCR entirely. Safe when the PDF has a clean text layer; saves memory.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Process the PDF N pages at a time (e.g. 5 or 10). Works around std::bad_alloc.")
    ap.add_argument("--skip-pages", type=str, default=None,
                    help="Skip pages that cause std::bad_alloc, e.g. '21-33' or '21,22,23,40'.")
    args = ap.parse_args()
    parse(
        PDF_PATH,
        DATA_DIR,
        force_full_ocr=args.force_full_ocr,
        do_ocr=not args.no_ocr,
        batch_size=args.batch_size,
        skip_pages=_parse_skip_pages(args.skip_pages),
    )


if __name__ == "__main__":
    main()
