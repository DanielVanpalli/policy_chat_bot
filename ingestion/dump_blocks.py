"""
Dump every text block and table from dataset/parsed.json to a flat .txt
file, separated by '############'. Useful for eyeballing what Docling
extracted before the real chunker is built.

Each block is prefixed with a small header showing page, label, and
(for tables) dimensions.

Usage:
    python -m ingestion.dump_blocks
    python -m ingestion.dump_blocks --in dataset/parsed.json --out dataset/blocks.txt
"""
import argparse
import json
from pathlib import Path


SEP = "\n" + "#" * 12 + "\n"


def _page_of(item: dict) -> str:
    prov = item.get("prov") or []
    if not prov:
        return "?"
    return str(prov[0].get("page_no", "?"))


def _table_to_markdown(tbl: dict) -> str:
    data = tbl.get("data", {})
    grid = data.get("grid") or []
    if not grid:
        return "[empty table]"
    lines = []
    for r, row in enumerate(grid):
        cells = [(c.get("text") or "").replace("\n", " ").strip() for c in row]
        lines.append("| " + " | ".join(cells) + " |")
        if r == 0:
            lines.append("|" + "|".join(["---"] * len(cells)) + "|")
    return "\n".join(lines)


def dump(in_path: Path, out_path: Path) -> None:
    doc = json.loads(in_path.read_text(encoding="utf-8"))
    texts = doc.get("texts", [])
    tables = doc.get("tables", [])

    blocks: list[tuple[tuple, str]] = []

    for t in texts:
        prov = t.get("prov") or [{}]
        page = prov[0].get("page_no", 0)
        charspan_start = prov[0].get("charspan", [0])[0]
        label = t.get("label", "text")
        header = f"[page {page} | {label}"
        if t.get("level") is not None:
            header += f" | level {t['level']}"
        header += "]"
        body = t.get("text", "") or ""
        blocks.append(((page, charspan_start, 0), f"{header}\n{body}"))

    for i, tbl in enumerate(tables):
        prov = tbl.get("prov") or [{}]
        page = prov[0].get("page_no", 0)
        data = tbl.get("data", {})
        nrows = data.get("num_rows", "?")
        ncols = data.get("num_cols", "?")
        header = f"[page {page} | table #{i} | {nrows}x{ncols}]"
        body = _table_to_markdown(tbl)
        blocks.append(((page, 10**9, 1), f"{header}\n{body}"))

    blocks.sort(key=lambda x: x[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(SEP.join(b[1] for b in blocks), encoding="utf-8")
    print(f"Wrote {len(blocks)} blocks -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, default=Path("dataset/parsed.json"))
    ap.add_argument("--out", type=Path, default=Path("dataset/blocks.txt"))
    args = ap.parse_args()
    dump(args.in_path, args.out)


if __name__ == "__main__":
    main()
