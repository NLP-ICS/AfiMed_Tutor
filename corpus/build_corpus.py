"""Corpus build pipeline (Phase 1).

Steps:
  1. Download guideline PDFs from sources.yaml (with fallback to mirror).
  2. Extract text using pdfplumber.
  3. Chunk using structure-aware chunker (primary) OR naive fixed-size chunker
     (for comparison baseline — pass --naive flag).
  4. Embed all chunks.
  5. Build FAISS IndexFlatIP and save index + metadata.

Usage:
    python corpus/build_corpus.py [--naive] [--dry-run]
    python corpus/build_corpus.py --max-chunk-tokens 200 --overlap-tokens 50
    python corpus/build_corpus.py --naive --out-dir artifacts/corpus/e1_naive

Environment (optional overrides for structure-aware naive limits):
    MAX_CHUNK_TOKENS  — defaults to CLI or 800
    OVERLAP_TOKENS    — defaults to CLI or 100
    EMBEDDING_MAX_INPUT_TOKENS — OpenAI/Voyage ceiling per text (default 8000; max 8192)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _REPO_ROOT / "corpus" / "raw"
SOURCES_PATH = _REPO_ROOT / "corpus" / "sources.yaml"
CHUNKS_PATH = _REPO_ROOT / "corpus" / "chunks.jsonl"
META_PATH = _REPO_ROOT / "corpus" / "chunks_meta.jsonl"
INDEX_PATH = _REPO_ROOT / "corpus" / "faiss.index"

MAX_CHUNK_TOKENS = 800
OVERLAP_TOKENS = 100

# ---------------------------------------------------------------------------
# Section-header regexes per guideline family (§6.1)
# ---------------------------------------------------------------------------

_HEADER_PATTERNS: dict[str, list[re.Pattern]] = {
    "ghana_stg": [
        re.compile(r"^\d+\.\s+[A-Z][A-Z\s/&,-]{3,}$", re.M),       # "1. MALARIA"
        re.compile(r"^[A-Z][A-Z\s/&,-]{4,}$", re.M),                # all-caps section
        re.compile(r"^(?:Diagnosis|Investigations|Management|Referral)\b", re.M | re.I),
    ],
    "sa_stg": [
        re.compile(r"^\d+\.\d+\s+[A-Z]", re.M),                     # "2.3 Hypertension"
        re.compile(r"^[A-Z][a-z ]+:\s*$", re.M),                    # "Diagnosis:"
        re.compile(r"^(?:CHAPTER|Section|Part)\s+\d+", re.M | re.I),
    ],
    "who_imci": [
        re.compile(r"^(?:ASSESS|CLASSIFY|TREAT|COUNSEL|FOLLOW-UP)\b", re.M | re.I),
        re.compile(r"^[A-Z][A-Z\s]{4,}$", re.M),
    ],
    "kenya_cg": [
        re.compile(r"^\d+\.\s+[A-Z]", re.M),
        re.compile(r"^(?:Diagnosis|Management|Treatment|Referral)\b", re.M | re.I),
    ],
}
# default fallback
_HEADER_PATTERNS["default"] = [re.compile(r"^\d+[\.\)]\s+[A-Z]", re.M)]


# ---------------------------------------------------------------------------
# Tokenisation (cheap approximation — avoids requiring a full tokeniser)
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    return len(text.split()) * 4 // 3  # rough chars→tokens


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------

def _split_at_paragraph_boundaries(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split a long text block at paragraph boundaries with overlap."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_tokens(para)
        if current_tokens + para_tokens > max_tokens and current_parts:
            chunks.append("\n\n".join(current_parts))
            # Keep overlap by retaining the last paragraph(s) that fit in overlap_tokens
            overlap_parts: list[str] = []
            overlap_count = 0
            for p in reversed(current_parts):
                t = _approx_tokens(p)
                if overlap_count + t <= overlap_tokens:
                    overlap_parts.insert(0, p)
                    overlap_count += t
                else:
                    break
            current_parts = overlap_parts
            current_tokens = overlap_count
        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))
    return chunks


def structure_aware_chunks(
    text: str,
    source_doc: str,
    family: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[dict]:
    """Two-pass structure-aware chunker (§6.1)."""
    patterns = _HEADER_PATTERNS.get(family, _HEADER_PATTERNS["default"])

    # First pass: split on detected section headers
    combined_pattern = re.compile(
        "|".join(f"(?:{p.pattern})" for p in patterns), re.M
    )
    matches = list(combined_pattern.finditer(text))

    sections: list[tuple[str, str]] = []  # (header, body)
    if not matches:
        sections.append(("(document)", text))
    else:
        # Text before first header
        if matches[0].start() > 0:
            sections.append(("(preamble)", text[: matches[0].start()]))
        for i, m in enumerate(matches):
            header = m.group(0).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            sections.append((header, body))

    # Second pass: split oversized sections at paragraph boundaries
    chunks: list[dict] = []
    chunk_idx = 0
    for section_title, body in sections:
        if _approx_tokens(body) <= max_tokens:
            sub_chunks = [body] if body else []
        else:
            sub_chunks = _split_at_paragraph_boundaries(body, max_tokens, overlap_tokens)

        for sub in sub_chunks:
            if not sub.strip():
                continue
            chunk_id = f"{source_doc}__{chunk_idx:05d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": sub.strip(),
                    "source_doc": source_doc,
                    "section_title": section_title,
                    "condition": _infer_condition(section_title),
                    "page_number": None,  # page tracking requires per-page extraction
                }
            )
            chunk_idx += 1
    return chunks


def naive_fixed_chunks_chars(
    text: str,
    source_doc: str,
    window_chars: int = 800,
    overlap_chars: int = 200,
) -> list[dict]:
    """Naive sliding-window chunker: fixed characters + overlap — no sections (E1 baseline)."""
    chunks: list[dict] = []
    if window_chars <= 0:
        return chunks
    overlap_chars = max(0, min(overlap_chars, window_chars - 1))
    stride = window_chars - overlap_chars
    if stride <= 0:
        stride = window_chars

    pos = 0
    ci = 0
    while pos < len(text):
        end = min(pos + window_chars, len(text))
        sub = text[pos:end]
        if sub.strip():
            chunks.append(
                {
                    "chunk_id": f"{source_doc}__naive_{ci:05d}",
                    "text": sub.strip(),
                    "source_doc": source_doc,
                    "section_title": "(none)",
                    "condition": None,
                    "page_number": None,
                }
            )
            ci += 1
        pos += stride

    return chunks


def naive_fixed_chunks(
    text: str,
    source_doc: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[dict]:
    """Word-window naive baseline for unit tests (historical).

    The CLI `--naive` flag uses naive_fixed_chunks_chars (800-char / 200-overlap per E1).
    """
    words = text.split()
    chunks: list[dict] = []
    step = max_tokens * 3 // 4
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_tokens * 3 // 4]
        if not chunk_words:
            continue
        chunk_text = " ".join(chunk_words)
        chunk_id = f"{source_doc}__naive_{len(chunks):05d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source_doc": source_doc,
                "section_title": "(none)",
                "condition": None,
                "page_number": None,
            }
        )
    return chunks


def _infer_condition(section_title: str) -> str | None:
    """Best-effort extraction of condition name from section title."""
    # Capitalised words that look like disease names
    m = re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", section_title)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_by_page(pdf_path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, page_text).

    Uses pypdf (fast) as the primary extractor; falls back to pdfplumber only
    if pypdf yields no text for a page (e.g. scanned pages with embedded fonts).
    """
    from pypdf import PdfReader

    pages: list[tuple[int, str]] = []
    try:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append((i, text))
        total_chars = sum(len(t) for _, t in pages)
        log.info("  pypdf extracted %d chars across %d pages", total_chars, len(pages))
        if total_chars > 0:
            return pages
    except Exception as e:
        log.warning("pypdf failed for %s: %s — trying pdfplumber", pdf_path.name, e)

    # Fallback: pdfplumber (slower but handles more layouts)
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                pages.append((i, text))
        return pages
    except Exception as e:
        log.error("pdfplumber also failed for %s: %s", pdf_path.name, e)
        return []


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, timeout: int = 120) -> bool:
    try:
        log.info("Downloading %s → %s", url, dest.name)
        resp = requests.get(url, timeout=timeout, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        log.info("  Saved %d bytes.", len(resp.content))
        return True
    except Exception as e:
        log.warning("  Download failed: %s", e)
        return False


def download_sources(sources: list[dict]) -> dict[str, Path | None]:
    """Download all PDFs; return mapping source_id → local path (None if failed)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path | None] = {}
    for src in sources:
        sid = src["id"]
        dest = RAW_DIR / f"{sid}.pdf"
        if dest.exists():
            log.info("Already cached: %s", dest.name)
            result[sid] = dest
            continue
        ok = _download(src["primary_url"], dest)
        if not ok and src.get("mirror_url"):
            log.info("Trying mirror for %s", sid)
            ok = _download(src["mirror_url"], dest)
        if ok:
            result[sid] = dest
        else:
            log.warning("SKIPPING %s — both primary and mirror failed.", sid)
            result[sid] = None
    return result


# ---------------------------------------------------------------------------
# Embedding input budget — OpenAI / Voyage reject inputs >8192 tokens
# ---------------------------------------------------------------------------

_EMBED_ENCODING_NAME = "cl100k_base"  # OpenAI text-embedding-* family


def _split_text_at_token_budget(text: str, enc, max_tokens: int) -> list[str]:
    """Split `text` into pieces each with at most `max_tokens` tokenizer tokens."""
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        t = text.strip()
        return [t] if t else []
    parts: list[str] = []
    for start in range(0, len(ids), max_tokens):
        chunk_ids = ids[start : start + max_tokens]
        decoded = enc.decode(chunk_ids).strip()
        if decoded:
            parts.append(decoded)
    return parts


def enforce_embedding_input_budget(chunks: list[dict]) -> list[dict]:
    """Ensure each chunk fits the active embedding API's maximum input length.

    PDF extraction can yield huge space-free spans that evade heuristic chunk sizing;
    tiktoken catches those before embedding.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    if provider not in {"openai", "voyage"}:
        return chunks

    try:
        import tiktoken
    except ImportError:
        log.warning(
            "tiktoken not installed — cannot enforce embedding input token ceiling; "
            "install tiktoken or set EMBEDDING_PROVIDER=local."
        )
        return chunks

    raw_budget = os.getenv("EMBEDDING_MAX_INPUT_TOKENS", "8000")
    max_tokens = int(raw_budget)
    max_tokens = max(256, min(max_tokens, 8192))

    enc = tiktoken.get_encoding(_EMBED_ENCODING_NAME)
    out: list[dict] = []
    oversized_fixed = 0

    for c in chunks:
        text = c.get("text") or ""
        if not text.strip():
            continue
        pieces = _split_text_at_token_budget(text, enc, max_tokens)
        if len(pieces) <= 1:
            if pieces:
                normalized = dict(c)
                normalized["text"] = pieces[0]
                out.append(normalized)
            continue
        oversized_fixed += 1
        base_id = str(c["chunk_id"])
        for i, piece in enumerate(pieces):
            row = dict(c)
            row["chunk_id"] = f"{base_id}__emb{i:03d}"
            row["text"] = piece
            out.append(row)

    if oversized_fixed:
        log.warning(
            "Split %d oversized chunks using %s tokenizer (≤%d tokens per embed for %s provider)",
            oversized_fixed,
            _EMBED_ENCODING_NAME,
            max_tokens,
            provider,
        )
    log.info(
        "Chunks eligible for embedding: %d raw chunk records → %d after API budget enforcement",
        len(chunks),
        len(out),
    )
    return out


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(texts: list[str]) -> list[list[float]]:
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    batch_size = 100
    embeddings: list[list[float]] = []

    if provider == "local":
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        log.info("Loading local embedding model: %s", model_name)
        model = SentenceTransformer(model_name)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            embeddings.extend(v.tolist() for v in vecs)
            log.info("  Embedded %d/%d chunks", min(i + batch_size, len(texts)), len(texts))
        return embeddings

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if provider == "openai":
            import openai
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
            embeddings.extend(item.embedding for item in resp.data)
        elif provider == "voyage":
            import voyageai
            vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
            result = vo.embed(batch, model="voyage-3-lite", input_type="document")
            embeddings.extend(result.embeddings)
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")
        log.info("  Embedded %d/%d chunks", min(i + batch_size, len(texts)), len(texts))
    return embeddings


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: list[list[float]]) -> "faiss.Index":
    import faiss
    import numpy as np

    vecs = np.array(embeddings, dtype="float32")
    # Normalise for cosine similarity via inner product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms > 0, norms, 1)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_output_paths(out_dir: Path | None) -> tuple[Path, Path, Path]:
    if out_dir is None:
        return CHUNKS_PATH, META_PATH, INDEX_PATH
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "chunks.jsonl", out_dir / "chunks_meta.jsonl", out_dir / "faiss.index"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AfriMed Tutor corpus")
    parser.add_argument("--naive", action="store_true", help="Use naive char-window chunker (800 chars, 200 overlap)")
    parser.add_argument(
        "--naive-window-chars",
        type=int,
        default=800,
        metavar="N",
        help="Character window size for --naive (default: 800)",
    )
    parser.add_argument(
        "--naive-overlap-chars",
        type=int,
        default=200,
        metavar="N",
        help="Overlap in characters for --naive (default: 200)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip embedding and indexing")
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Max approximate tokens per chunk for structure-aware chunker "
        "(or env MAX_CHUNK_TOKENS; default: 800)",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Overlap tokens for intra-section splitting "
        "(or env OVERLAP_TOKENS; default: 100)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write chunks.jsonl, chunks_meta.jsonl, and faiss.index here "
        "(default: corpus/ under repo root)",
    )
    args = parser.parse_args()

    max_tokens = args.max_chunk_tokens
    if max_tokens is None:
        max_tokens = int(os.getenv("MAX_CHUNK_TOKENS", MAX_CHUNK_TOKENS))
    overlap_tokens = args.overlap_tokens
    if overlap_tokens is None:
        overlap_tokens = int(os.getenv("OVERLAP_TOKENS", OVERLAP_TOKENS))

    chunks_out, meta_out, index_out = _resolve_output_paths(args.out_dir)

    with open(SOURCES_PATH) as f:
        sources = yaml.safe_load(f)["sources"]

    downloaded = download_sources(sources)

    all_chunks: list[dict] = []
    source_family = {s["id"]: s["family"] for s in sources}

    for src in sources:
        sid = src["id"]
        pdf_path = downloaded.get(sid)
        if pdf_path is None:
            log.warning("No PDF for %s, skipping.", sid)
            continue

        log.info("Extracting text from %s", pdf_path.name)
        pages = extract_text_by_page(pdf_path)
        full_text = "\n".join(text for _, text in pages)

        if args.naive:
            chunks = naive_fixed_chunks_chars(
                full_text,
                source_doc=sid,
                window_chars=args.naive_window_chars,
                overlap_chars=args.naive_overlap_chars,
            )
        else:
            chunks = structure_aware_chunks(
                full_text,
                source_doc=sid,
                family=source_family.get(sid, "default"),
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )

        # Attach page numbers where possible (best-effort: use page index as hint)
        log.info("  %d chunks from %s", len(chunks), sid)
        all_chunks.extend(chunks)

    log.info("Total chunks (pre embedding-budget): %d", len(all_chunks))
    all_chunks = enforce_embedding_input_budget(all_chunks)

    if args.dry_run:
        log.info("Dry-run: skipping embedding and indexing.")
        return

    # Save metadata
    meta_out.write_text("\n".join(json.dumps(c) for c in all_chunks), encoding="utf-8")
    chunks_out.write_text(
        "\n".join(json.dumps({"chunk_id": c["chunk_id"], "text": c["text"]}) for c in all_chunks),
        encoding="utf-8",
    )

    # Embed and index
    texts = [c["text"] for c in all_chunks]
    log.info("Embedding %d chunks...", len(texts))
    embeddings = embed_chunks(texts)

    import faiss
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(index_out))
    log.info("FAISS index saved to %s (%d vectors, dim=%d)", index_out, index.ntotal, index.d)


if __name__ == "__main__":
    main()
