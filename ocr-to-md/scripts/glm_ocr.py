#!/usr/bin/env python3
"""Run local Ollama glm-ocr on a full notebook page and house-style the Markdown.

All processing stays on this Mac (Ollama + Apple Vision hints + /usr/share/dict).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

OLLAMA = "http://127.0.0.1:11434/api/generate"
TAGS = "http://127.0.0.1:11434/api/tags"
OCR_MODEL_DEFAULT = "glm-ocr"
POLISH_MODELS = ("llama3.2:3b", "llama3.2:1b", "qwen2.5:1.5b", "gemma2:2b")
MAC_DICT = Path("/usr/share/dict/words")


def total_ram_gb() -> float:
    try:
        raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(raw) / (1024 ** 3)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 16.0


LOW_MEM = os.environ.get("OCR_LOW_MEM") == "1" or total_ram_gb() <= 8.5
KEEP_ALIVE = os.environ.get("OCR_KEEP_ALIVE") or ("30m" if LOW_MEM else "8m")
# 8GB M1: glm-ocr:q8_0 is 1.6GB. 4k ctx was too small for a full page (vision
# tokens ≈ pixels/784), which forced tiny tiles that scrambled handwriting.


def _int_env(name: str, default: int, *, min_value: int | None = None) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    if min_value is not None and value < min_value:
        return default
    return value


NUM_CTX = _int_env("OCR_NUM_CTX", 10240 if LOW_MEM else 12288, min_value=8192)
FIRST_SHRINK = _int_env("OCR_TILE_MAX", 1792 if LOW_MEM else 2048, min_value=1536)
NUM_PREDICT = _int_env("OCR_NUM_PREDICT", 4096 if LOW_MEM else 6144, min_value=2048)

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K\r"


def cprint(msg: str, *, color: str = "") -> None:
    if sys.stderr.isatty() and color:
        sys.stderr.write(f"{color}{msg}{RESET}\n")
    else:
        sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def progress(msg: str) -> None:
    if sys.stderr.isatty():
        sys.stderr.write(f"{CLEAR_LINE}{CYAN}│{RESET} {msg}")
        sys.stderr.flush()
    else:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()


def progress_done(msg: str) -> None:
    if sys.stderr.isatty():
        sys.stderr.write(f"{CLEAR_LINE}{GREEN}│{RESET} {msg}\n")
    else:
        sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# Official glm-ocr task prefix. Long instruction dumps make the 0.5B decoder
# echo the prompt and shuffle crops; Cursor-quality OCR is the whole page + this.
TEXT_PROMPT = """Text Recognition:"""

TEXT_PROMPT_COLUMNS = """Text Recognition:
This notebook page has side-by-side columns. Read the whole page. Keep column order. Markdown only."""

TEXT_PROMPT_CONTINUE = """Text Recognition:
This is the lower part of the same notebook page. Transcribe every remaining line. Markdown only."""

COLUMN_PROMPT = TEXT_PROMPT
BOX_PROMPT = """Box Recognition:"""
TABLE_PROMPT = """Table Recognition:"""
FIGURE_PROMPT = """Figure Recognition:"""


def ollama_generate(
    *,
    model: str,
    prompt: str | None = None,
    images: list[str] | None = None,
    timeout: int = 240,
    keep_alive: str | int | None = None,
    num_predict: int = 2048,
) -> tuple[str, dict]:
    payload: dict = {
        "model": model,
        "stream": False,
        "keep_alive": keep_alive if keep_alive is not None else KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": NUM_CTX,
            "num_gpu": 99,
            "repeat_penalty": 1.05,
        },
    }
    if prompt is not None:
        payload["prompt"] = prompt
    if images:
        payload["images"] = images
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as exc:
        raise SystemExit(f"Ollama {model} failed: {exc}") from exc
    text = (data.get("response") or "").strip()
    meta = {
        "done": bool(data.get("done", True)),
        "done_reason": str(data.get("done_reason") or ""),
        "eval_count": int(data.get("eval_count") or 0),
        "num_predict": num_predict,
    }
    return text, meta


def ollama_unload(model: str) -> None:
    """Drop a model from VRAM without running a dummy generation."""
    payload = {"model": model, "keep_alive": 0}
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        pass


def image_max_side(path: Path) -> int:
    try:
        out = subprocess.check_output(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return FIRST_SHRINK
    width = height = 0
    for line in out.splitlines():
        if "pixelWidth:" in line:
            width = int(line.split(":")[-1].strip() or 0)
        elif "pixelHeight:" in line:
            height = int(line.split(":")[-1].strip() or 0)
    return max(width, height)


def shrink_jpeg(src: Path, max_side: int) -> Path:
    dest = src.with_name(f"{src.stem}.s{max_side}.jpg")
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime and dest.stat().st_size > 1000:
        return dest
    subprocess.run(
        ["sips", "-Z", str(max_side), "-s", "format", "jpeg", str(src), "--out", str(dest)],
        check=True,
        capture_output=True,
    )
    return dest


def ollama_ocr(image_path: Path, prompt: str, timeout: int = 720) -> tuple[str, dict]:
    """Send the full page to local glm-ocr. Shrink only if the JPEG is larger than the budget."""
    send = image_path
    if image_max_side(image_path) > FIRST_SHRINK:
        send = shrink_jpeg(image_path, FIRST_SHRINK)
    last_err = ""
    tried: set[str] = set()
    shrink_ladder = (None, FIRST_SHRINK, 1536, 1280) if LOW_MEM else (None, FIRST_SHRINK, 1600)
    n_pred = NUM_PREDICT
    last_meta: dict = {}
    for max_side in shrink_ladder:
        if max_side is not None:
            send = shrink_jpeg(image_path, max_side)
        key = str(send)
        if key in tried:
            continue
        tried.add(key)
        raw = send.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        del raw
        try:
            out, meta = ollama_generate(
                model=ocr_model(),
                prompt=prompt,
                images=[b64],
                timeout=timeout,
                keep_alive=KEEP_ALIVE,
                num_predict=n_pred,
            )
            last_meta = meta
            return strip_instruction_leaks(out), meta
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            last_err = f"HTTP {exc.code}: {body}"
            progress(f"retry {image_path.name} smaller (HTTP {exc.code})…")
        except urllib.error.URLError as exc:
            last_err = str(exc)
            progress(f"retry {image_path.name} ({exc})…")
        except subprocess.CalledProcessError as exc:
            last_err = str(exc)
        finally:
            del b64
    raise SystemExit(f"Ollama {ocr_model()} failed after retries: {last_err or last_meta}")


def list_models() -> set[str]:
    try:
        with urllib.request.urlopen(TAGS, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return set()
    names: set[str] = set()
    for m in data.get("models") or []:
        name = m.get("name") or ""
        if name:
            names.add(name)
            names.add(name.split(":")[0])
    return names


@lru_cache(maxsize=1)
def ocr_model() -> str:
    env = (os.environ.get("OCR_GLM_MODEL") or "").strip()
    if env:
        return env
    names = list_models()
    q8 = [n for n in names if n.startswith("glm-ocr") and "q8" in n]
    if LOW_MEM and q8:
        return q8[0]
    if "glm-ocr:q8_0" in names and "glm-ocr:latest" not in names and "glm-ocr" not in names:
        return "glm-ocr:q8_0"
    if "glm-ocr" in names or "glm-ocr:latest" in names:
        return "glm-ocr"
    tagged = sorted(n for n in names if n.startswith("glm-ocr"))
    return tagged[0] if tagged else OCR_MODEL_DEFAULT


def pick_polish_model() -> str | None:
    names = list_models()
    for candidate in POLISH_MODELS:
        if candidate in names:
            return candidate
        for n in names:
            if n.startswith(candidate.split(":")[0] + ":"):
                return n
    return None


def tile_kind(tile: dict) -> str:
    prompt = (tile.get("prompt") or "").lower()
    name = tile.get("file") or ""
    if "figure" in prompt or name.startswith("figure"):
        return "figure"
    if "box" in prompt or name.startswith("box"):
        return "box"
    if "table" in prompt or "table" in name:
        return "table"
    return "text"


def tile_column(tile: dict) -> str:
    col = (tile.get("column") or "").lower()
    if col in {"left", "right", "mid", "full", "figure", "box"}:
        return col
    name = tile.get("file") or ""
    if name.startswith("col-L") or "-L." in name or name.endswith("-L.jpg"):
        return "left"
    if name.startswith("col-R") or "-R." in name or name.endswith("-R.jpg"):
        return "right"
    if "-M." in name or name.endswith("-M.jpg"):
        return "mid"
    if name.startswith("figure"):
        return "figure"
    if name.startswith("box"):
        return "box"
    return "full"


def tile_zone(tile: dict) -> int:
    if "zone" in tile:
        try:
            return int(tile["zone"])
        except (TypeError, ValueError):
            pass
    name = tile.get("file") or ""
    m = re.match(r"z(\d+)", name)
    return int(m.group(1)) if m else 0


def prompt_for(tile: dict, page_columns: int = 1) -> str:
    kind = tile_kind(tile)
    col = tile_column(tile)
    if kind == "figure":
        return FIGURE_PROMPT
    if kind == "box" or col == "box":
        return BOX_PROMPT
    if kind == "table":
        return TABLE_PROMPT
    if tile.get("continuation"):
        return TEXT_PROMPT_CONTINUE
    if page_columns >= 2:
        return TEXT_PROMPT_COLUMNS
    return TEXT_PROMPT


def norm_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", norm_line(s)) if len(t) > 2}


_MODEL_LEAK = re.compile(
    r"""(?ix)^\s*[-*>]*\s*(
        use\s+(latex|markdown|html|katex|hem[l])\b
        |output\s+clean\s+markdown\b
        |this\s+crop\s+is\s+one\s+column\b
        |read\s+only\s+this\s+column\b
        |underlined\s+(text|words)\s*→
        |bold\s*/\s*heavy\s+ink
        |titles\s*→
        |if\s+this\s+column\s+contains\s+a\s+drawn\s+box
        |lists,\s*math,\s*and\s+paragraph
        |blank\s+line\s+wherever
        |transcribe\s+every\s+line
        |never\s+summarize
        |fix\s+messy\s+handwriting\s+using\s+the\s+sentence
        |do\s+not\s+invent\s+content
        |markdown\s+only
        |no\s+h[te]ml\s+divs
        |text\s+recognition:
        |column\s+recognition:
        |box\s+recognition:
        |table\s+recognition:
        |figure\s+recognition:
        |ignore\s+a\s+sliver
        |emphasis\s+on\s+(math|formulas|layout)
        |final\s+check\s+of\s+the\s+text
        |write\s+only\s+the\s+transcribed
        |never\s+repeat\s+or\s+quote
        |if\s+possible\s*[.!]?\s*$
        |the\s+student'?s\s+handwriting
        |the\s+instructions\s+are\s+very
        |notes\s+are\s+(organized|written)\s+in
        |use\s+a\s+(standard|clear|consistent|legible)\s+font
        |use\s+a\s+consistent\s+font\s+(size|color|style)
    )
    """,
)

_LATEX_LOOP = re.compile(r"(\\square|\\triangle|\\diamond|\\frown)", flags=re.I)


def looks_like_notes(line: str) -> bool:
    """Real worksheet sentences must never be treated as prompt echo."""
    words = re.findall(r"[A-Za-z']{3,}", line)
    if len(words) < 5:
        return False
    lex = lexicon()
    hits = sum(1 for w in words if w.lower() in lex or _looks_inflected(w.lower(), lex))
    return hits >= 4 and hits / len(words) >= 0.55


def is_model_leak(line: str) -> bool:
    raw = line.strip()
    s = re.sub(r"^[-*>\s]+", "", raw)
    s = re.sub(r"</?u>", "", s, flags=re.I)
    if not s:
        return False
    if _MODEL_LEAK.match(s) or _MODEL_LEAK.match(raw):
        return True
    if s.lower().startswith("use latex") or "use latex for" in s.lower():
        return True
    if _LATEX_LOOP.findall(s) and len(_LATEX_LOOP.findall(s)) >= 3:
        return True
    if len(s) > 280 and s.count("\\") >= 6:
        return True
    # Remaining heuristics are easy to over-match — keep real notes.
    if looks_like_notes(s):
        return False
    if re.match(r"^#{1,6}\s+(text|markdown|output|notes only)\s*$", s, flags=re.I):
        return True
    if re.sub(r"[#*_`\s]+", "", s).lower() in {"text", "markdown", "output", "sketch"}:
        return True
    return False


def strip_instruction_leaks(text: str) -> str:
    """Drop echoed model instructions without touching real notes."""
    kept: list[str] = []
    for raw in text.splitlines():
        cut = re.split(r"(?i)\s+-\s+use\s+latex\b", raw, maxsplit=1)[0].rstrip()
        if is_model_leak(cut) or is_model_leak(raw):
            continue
        kept.append(cut)
    return "\n".join(kept)


def is_duplicate(line: str, recent: list[str]) -> bool:
    n = norm_line(line)
    if not n:
        return True
    if is_model_leak(line):
        return True
    if n in {
        "text recognition:", "table recognition:", "figure recognition:",
        "column recognition:", "box recognition:",
    }:
        return True
    if n.startswith("here is") or n.startswith("sure,"):
        return True
    for prev in recent:
        p = norm_line(prev)
        if n == p:
            return True
        shorter, longer = (n, p) if len(n) <= len(p) else (p, n)
        if len(shorter) >= 24 and shorter in longer and len(shorter) / len(longer) >= 0.88:
            return True
        a, b = token_set(n), token_set(p)
        if a and b and min(len(a), len(b)) >= 6 and len(a & b) / len(a | b) >= 0.94:
            return True
    return False


def html_table_to_md(html: str) -> str:
    rows: list[list[str]] = []
    for tr in re.findall(r"<tr\b[^>]*>(.*?)</tr>", html, flags=re.I | re.S):
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr, flags=re.I | re.S)
        cleaned = [re.sub(r"<[^>]+>", "", c) for c in cells]
        cleaned = [re.sub(r"\s+", " ", c).strip() for c in cleaned]
        cleaned = [
            c.replace("&#x27;", "'").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            for c in cleaned
        ]
        if any(cleaned):
            rows.append(cleaned)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    if width < 2:
        return ""
    rich = sum(1 for r in rows if sum(1 for c in r if c.strip()) >= 2)
    if rich < 2:
        return ""
    normed = [r + [""] * (width - len(r)) for r in rows]
    header, body = normed[0], normed[1:]
    if all(re.sub(r"[-:\s]", "", c) == "" for c in header):
        return ""
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def extract_structured(text: str) -> str:
    tables = [
        html_table_to_md(t)
        for t in re.findall(r"<table\b[^>]*>.*?</table>", text, flags=re.I | re.S)
    ]
    tables = [t for t in tables if t]
    leftover = re.sub(r"<table\b[^>]*>.*?</table>", "\n", text, flags=re.I | re.S)
    leftover = leftover.replace("```markdown", "").replace("```", "").strip()
    parts = [p for p in [leftover, *tables] if p]
    return "\n\n".join(parts)


def looks_like_diagram(text: str) -> bool:
    low = text.lower()
    keys = (
        "rain", "arrow", "soil", "diagram", "co2", "o2", "hydrosphere",
        "geosphere", "figure", "labeled", "shows", "cycle",
    )
    if not any(k in low for k in keys):
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) >= 3


def stitch_lines(texts: list[str]) -> str:
    out: list[str] = []
    recent: list[str] = []
    for text in texts:
        text = extract_structured(text)
        kept: list[str] = []
        for ln in [line.rstrip() for line in text.splitlines()]:
            if is_duplicate(ln, recent[-16:]):
                continue
            kept.append(ln)
            if ln.strip():
                recent.append(ln)
        blob = "\n".join(kept).strip()
        if blob:
            out.append(blob)
    return "\n\n".join(out).strip()


def md_quote_block(text: str, prefix: str = "> ") -> str:
    lines = text.replace("\r\n", "\n").strip("\n").split("\n")
    out: list[str] = []
    for line in lines:
        out.append(prefix.rstrip() if line.strip() == "" else prefix + line)
    return "\n".join(out)


def two_col_md(left: str, right: str) -> str:
    """Side-by-side columns as Obsidian callouts (no raw HTML in the note)."""
    return (
        "> [!columns]\n"
        ">\n"
        "> > [!left]\n"
        f"{md_quote_block(left.strip(), '> > ')}\n"
        ">\n"
        "> > [!right]\n"
        f"{md_quote_block(right.strip(), '> > ')}\n"
    )


def three_col_md(left: str, mid: str, right: str) -> str:
    return (
        "> [!columns3]\n"
        ">\n"
        "> > [!left]\n"
        f"{md_quote_block(left.strip(), '> > ')}\n"
        ">\n"
        "> > [!mid]\n"
        f"{md_quote_block(mid.strip(), '> > ')}\n"
        ">\n"
        "> > [!right]\n"
        f"{md_quote_block(right.strip(), '> > ')}\n"
    )


def box_md(body: str, title: str = "") -> str:
    """Full-width worksheet frame — only for text that was boxed on the page."""
    inner = body.strip()
    if inner.startswith(">") and "[!box]" in inner.split("\n", 1)[0]:
        return inner
    head = f"> [!box] {title.strip()}" if title.strip() else "> [!box]"
    return f"{head}\n{md_quote_block(inner, '> ')}"


BOX_FENCE_RE = re.compile(r"\[BOX\](.*?)\[/BOX\]", flags=re.I | re.S)


def normalize_boxes(text: str) -> str:
    text = BOX_FENCE_RE.sub(lambda m: "\n" + box_md(m.group(1).strip()) + "\n", text)
    return text


OCR_COLS_RE = re.compile(
    r'<div class="ocr-cols"[^>]*>\s*'
    r"<div[^>]*>\s*(.*?)\s*</div>\s*"
    r"<div[^>]*>\s*(.*?)\s*</div>\s*"
    r"</div>",
    flags=re.I | re.S,
)


def html_cols_to_callouts(text: str) -> str:
    """Turn leftover flexbox HTML into readable callouts."""
    text = OCR_COLS_RE.sub(
        lambda m: two_col_md(m.group(1).strip(), m.group(2).strip()),
        text,
    )
    text = re.sub(
        r'<span style="color:[^"]+">([^<]+)</span>',
        r"==\1==",
        text,
        flags=re.I,
    )
    return text


_SPAN_LINE = re.compile(
    r"(?i)^\s*(?:\*\*)?(?:key\s*idea|main\s*idea|big\s*idea|essential\s*question)\b"
)


def lift_spanning_lines(left: str, mid: str, right: str) -> tuple[str, str, str, str]:
    """Pull page-wide lines (Key Idea) out of shredded columns."""
    chunks = {"left": left, "mid": mid, "right": right}
    lifted: list[str] = []

    def keep(side: str) -> str:
        kept: list[str] = []
        for raw in chunks[side].splitlines():
            s = re.sub(r"^[#>*\s]+", "", raw).strip()
            s = re.sub(r"[*_]+", "", s)
            if _SPAN_LINE.match(s) or (s.lower().startswith("key idea") and "niche" in s.lower()):
                lifted.append(s)
                continue
            kept.append(raw)
        return "\n".join(kept).strip()

    return keep("left"), keep("mid"), keep("right"), "\n".join(dict.fromkeys(lifted))


_COL_RANK = {"full": 0, "left": 1, "mid": 2, "right": 3, "box": 4, "figure": 5}


def _tile_bottom(item: dict) -> int:
    return int(item["y"]) + max(1, int(item["h"]))


def _cluster_reading_bands(items: list[dict]) -> list[list[dict]]:
    """Group tiles that sit on the same vertical slice, top to bottom."""
    ordered = sorted(
        items,
        key=lambda it: (int(it["y"]), _COL_RANK.get(it["column"], 9), int(it["x"])),
    )
    bands: list[list[dict]] = []
    for it in ordered:
        if not bands:
            bands.append([it])
            continue
        prev = bands[-1]
        prev_bot = max(_tile_bottom(p) for p in prev)
        # Overlap (or nearly) means the same row of the notebook, not the next section.
        if int(it["y"]) < prev_bot - 24:
            prev.append(it)
        else:
            bands.append([it])
    return bands


def stitch_band(sections: list[dict]) -> str:
    full: list[str] = []
    left: list[str] = []
    mid: list[str] = []
    right: list[str] = []
    boxes: list[str] = []
    figures: list[str] = []
    for item in sorted(
        sections,
        key=lambda it: (_COL_RANK.get(it["column"], 9), int(it["x"])),
    ):
        text = normalize_boxes(item["text"])
        kind = item["kind"]
        column = item["column"]
        if kind == "figure":
            blob = extract_structured(text)
            if looks_like_diagram(blob):
                figures.append(blob)
            elif blob:
                full.append(blob)
            continue
        if kind == "box" or column == "box":
            blob = extract_structured(text)
            if blob:
                boxes.append(blob)
            continue
        if column == "left":
            left.append(text)
        elif column == "mid":
            mid.append(text)
        elif column == "right":
            right.append(text)
        else:
            full.append(text)
    parts: list[str] = []
    head = stitch_lines(full)
    if head:
        parts.append(format_layout(head))
    left_body, mid_body, right_body = stitch_lines(left), stitch_lines(mid), stitch_lines(right)
    left_body, mid_body, right_body, lifted = lift_spanning_lines(left_body, mid_body, right_body)
    if lifted:
        parts.append(format_layout(lifted))
    if left_body and mid_body and right_body:
        parts.append(three_col_md(format_layout(left_body), format_layout(mid_body), format_layout(right_body)))
    elif left_body and right_body:
        parts.append(two_col_md(format_layout(left_body), format_layout(right_body)))
    elif left_body or mid_body or right_body:
        parts.append(format_layout(left_body or mid_body or right_body))
    for blob in boxes:
        laid = format_layout(blob)
        parts.append(laid if "[!box]" in laid else box_md(laid))
    body = "\n\n".join(p for p in parts if p).strip()
    if figures:
        body = f"{body}\n\n#### <u>**Diagram**</u>\n\n" + "\n\n".join(figures)
    return body


def stitch(sections: list[tuple]) -> str:
    """Assemble tiles in page reading order (top→bottom, then left→right)."""
    items: list[dict] = []
    for sec in sections:
        if len(sec) >= 7:
            kind, column, text, _zone, y, x, h = sec[:7]
        elif len(sec) >= 4:
            kind, column, text, _zone = sec[:4]
            y, x, h = 0, 0, 1
        else:
            continue
        if not (text or "").strip():
            continue
        items.append({
            "kind": kind,
            "column": column,
            "text": text,
            "y": int(y or 0),
            "x": int(x or 0),
            "h": int(h or 1),
        })
    parts = [stitch_band(band) for band in _cluster_reading_bands(items)]
    return "\n\n".join(p for p in parts if p).strip()


def colorize(text: str, red_words: list[str]) -> str:
    skip = {"this", "that", "with", "from", "they", "have", "been", "here", "stop"}
    words = sorted(
        {w for w in red_words if len(w) >= 4 and w.lower() not in skip},
        key=len,
        reverse=True,
    )
    for w in words:
        text = re.sub(
            rf"(?<!==)(?<!\w)({re.escape(w)})(?!\w)(?!==)",
            r"==\1==",
            text,
            flags=re.IGNORECASE,
        )
    if re.search(r"four\s+spheres|geosphere|hydrosphere", text, flags=re.I):
        text = re.sub(
            r"(?<!\w)(Four Spheres|Geosphere|Hydrosphere|Atmosphere|Biosphere)(?!\w)",
            r"**\1**",
            text,
        )
    return text


# --- local spelling (dictionary + page context; no network) -----------------

SCHOOL_VOCAB = {
    "rome", "roman", "romans", "greece", "greek", "greeks", "stoics", "stoic",
    "praetors", "praetor", "republic", "republicanism", "citizenship", "consuls",
    "senate", "democracy", "legislature", "judicial", "natural", "law",
    "homeostasis", "metabolism", "reproduction", "stimuli", "stimulus",
    "organism", "organisms", "population", "community", "ecosystem", "biosphere",
    "geosphere", "hydrosphere", "atmosphere", "photosynthesis", "symbiosis",
    "parasitism", "mutualism", "commensalism", "interspecific", "intraspecific",
    "predator", "prey", "evolution", "cellular", "genetic", "characteristics",
    "development", "organization", "inheritance", "generations", "abiotic",
    "biotic", "extraneous", "denominator", "numerator", "quadrant", "quadrants",
    "symmetry", "function", "functions", "equation", "equations", "linear",
    "vertical", "distance", "formula", "consecutive", "percent", "independent",
    "dependent", "variable", "variables", "dionysus", "maenads", "thyrsus",
    "philosophy", "literature", "enlightenment", "montesquieu", "holbach",
    "locke", "humanism", "influence", "influenced", "newton", "plato",
    "habitat", "habitats", "niche", "niches", "ecological", "abiotic",
    "biotic", "exponential", "logistic", "dispersion", "dispersed",
    "clumped", "uniform", "random", "density", "carrying", "capacity",
    "symbiosis", "mutualism", "parasitism", "commensalism", "reproduce",
    "biological", "dramatically", "crowded", "species", "environment",
    "temperature", "curve", "plateau", "plateaus", "fits",
    "predation", "competition", "drought", "wildfire", "volcanic",
    "eruptions", "hurricanes", "deforestation", "adequate", "limited",
    "limiting", "independent", "dependent",
}

OCR_WORD_FIXES = {
    "rorne": "Rome", "rorne's": "Rome's", "teh": "the", "adn": "and", "taht": "that",
    "recieve": "receive", "seperate": "separate", "occured": "occurred",
    "enviroment": "environment", "goverment": "government", "repubilc": "republic",
    "citizinship": "citizenship", "atmoshere": "atmosphere", "hydrospher": "hydrosphere",
    "biospher": "biosphere", "geospher": "geosphere", "ecosysten": "ecosystem",
    "organisim": "organism", "populaton": "population", "extraaneous": "extraneous",
    "extraneos": "extraneous", "demoninator": "denominator", "denomenator": "denominator",
    "symetry": "symmetry", "quadrent": "quadrant", "quadrents": "quadrants",
    "homeostatis": "homeostasis", "homeostais": "homeostasis", "mutalism": "mutualism",
    "mutaulism": "mutualism", "commensicism": "commensalism", "commensalism": "commensalism",
    "paratism": "parasitism", "parasitsm": "parasitism", "entaspecific": "intraspecific",
    "interpecifc": "interspecific", "photosythesis": "photosynthesis",
    "independant": "independent", "dependant": "dependent", "geneteration": "generations",
    "acteristics": "Characteristics", "nabolism": "Metabolism", "ostasis": "Homeostasis",
    "thyrpsos": "Thyrsus",
    "iafluenced": "influenced", "influenceu": "influenced", "influencd": "influenced",
    "inflenced": "influenced", "influense": "influence", "influance": "influence",
    "enlishtenment": "Enlightenment", "enlightment": "Enlightenment",
    "enlightenent": "Enlightenment", "enlghtenment": "Enlightenment",
    "scienthic": "Scientific", "revalution": "Revolution", "philosorhy": "philosophy",
    "greeik": "Greek", "greelk": "Greek", "nderlying": "underlying",
    "onstitution": "Constitution",
    "dispertion": "dispersion", "disperton": "dispersion", "disportion": "dispersion",
    "nantat": "habitat", "habital": "habitat", "habititat": "habitat",
    "abutic": "abiotic", "abotic": "abiotic", "biatie": "biotic",
    "ellagical": "ecological", "ecologial": "ecological",
    "biologich": "biological", "reproduçe": "reproduce", "reproduice": "reproduce",
    "surive": "survive", "envorment": "environment", "enviorment": "environment",
    "clumped": "clumped", "uniformal": "uniform", "unifrom": "uniform",
    "exponetial": "exponential", "experiential": "exponential",
    "logisitic": "logistic", "plateaus": "plateaus",
    "dits": "fits", "tood": "food", "toal": "food", "babitat": "habitat",
    "nantut": "habitat", "nantat": "habitat", "abilic": "abiotic",
    "speaies": "species", "speces": "species", "pateaus": "plateaus",
    "avenge": "average", "averge": "average", "partcular": "particular",
    "adequete": "adequate", "capezity": "capacity", "capeacity": "capacity",
    "thetter": "shelter", "ructors": "factors", "enriamental": "environmental",
    "popuation": "population", "compositions": "competition",
    "compertion": "competition", "priatio": "predation",
    "vocanic": "volcanic", "hutcanes": "hurricanes", "wilfire": "wildfire",
    "dought": "drought", "restation": "deforestation",
    "inuepenel": "independent", "tartarsi": "factors",
    "enviormental": "environmental", "enrismental": "environmental",
    "enviromental": "environmental", "carying": "carrying",
    "tactors": "factors", "tados": "factors", "indegendend": "independent",
    "indegendendent": "independent", "wicktre": "wildfire",
    "vacanic": "volcanic", "hurticanes": "hurricanes",
    "frostation": "deforestation", "comperton": "competition",
    "pratan": "predation", "polation": "population",
    "nupendent": "independent", "miting": "limiting",
    "predatation": "predation", "pedation": "predation",
    "limitited": "limited", "habertat": "habitat", "mates": "mates",
    "sizer": "size",
    "bitt": "habitat", "habitt": "habitat", "abitatt": "habitat",
    "organis": "organism", "organion": "organism",
    "bidogie": "biological", "bidegical": "biological", "bidlogical": "biological",
    "entiment": "environment", "envirment": "environment",
    "illation": "population", "chamatically": "dramatically",
    "relively": "relatively", "retirely": "relatively",
    "aroude": "crowded", "crated": "crowded",
    "dispertion": "dispersion", "disportion": "dispersion",
    "uniformal": "uniform", "clump": "clumped",
}

PHRASE_FIXES = (
    (r"\bha\s*\*?\*?bitt\b", "habitat"),
    (r"\*\*ha\*\*bitt", "habitat"),
    (r"\*\*dits\*\*", "fits"),
    (r"\bdits\b", "fits"),
    (r"\bVSS\b", "VS"),
    (r"\bcill\b", "all"),
    (r"dramatically a relatively", "dramatically over a relatively"),
    (r"increases dramatically a relatively", "increases dramatically over a relatively"),
    (r"Looks like a J\b(?!\s*-?\s*shaped)", "Looks like a J-shaped curve"),
    (r"\bJ shaped curve\b", "J-shaped curve"),
    (r"\bS shaped curve\b", "S-shaped curve"),
    (r"\bS-shaped curvei\b", "S-shaped curve"),
    (r"then plateaus into a\s*$", "then plateaus into an S-shaped curve"),
    (r"the \*\*m\*\*\. plateaus", "then plateaus"),
    (r"\bthe m\. plateaus\b", "then plateaus"),
    (r"total arec\s*km\s*2", "total area (km²)"),
    (r"total areckm\s*2", "total area (km²)"),
    (r"\bcommon demon\b", "common denominator"),
    (r"\b([Hh]ow did .{8,90}?)influenced\b", r"\1influence"),
    (r"\bdefend on\b", "depend on"),
    (r"\bdependend on\b", "depend on"),
    (r"\bDensity Dependent limiting tactors\b", "Density Dependent limiting factors"),
    (r"\bDensity Indegendent miting tados\b", "Density Independent limiting factors"),
    (r"\baverage population size in a particular\b", "average population size in a particular"),
    (r"\bcan changee\b", "can change"),
    (r"\bcan chang\b", "can change"),
    (r"\bCarrying Capezity\b", "Carrying Capacity"),
    (r"\blimited by\b", "limited by"),
    (r"\bvolcanic eruptions\b", "volcanic eruptions"),
    (r"limiting tartarsiAffects", "limiting factors: Affects"),
    (r"miting tartarsiAffects", "limiting factors: Affects"),
    (r"\binuepenel or por\b", "independent of population size"),
    (r"\bindependent or por\b", "independent of population size"),
    (r"\bDought,\s*wilfire\b", "Drought, wildfire"),
    (r"\ban de restation\b", "and deforestation"),
    (r"\bsizelcomportions priatio\b", "size (competition, predation"),
)

CONFUSIONS = (
    ("rn", "m"), ("m", "rn"), ("nn", "m"), ("cl", "d"), ("vv", "w"),
    ("ii", "u"), ("ri", "n"), ("in", "m"), ("ul", "d"), ("li", "h"),
    ("c", "e"), ("e", "c"), ("a", "o"), ("u", "n"), ("n", "u"),
    ("i", "l"), ("l", "i"), ("h", "b"),
)


@lru_cache(maxsize=1)
def lexicon() -> set[str]:
    words = {w.lower() for w in SCHOOL_VOCAB}
    if MAC_DICT.exists():
        for line in MAC_DICT.read_text(errors="ignore").splitlines():
            w = line.strip().lower()
            if w.isalpha() and 3 <= len(w) <= 28:
                words.add(w)
    return words


def apply_ocr_word_fixes(text: str) -> str:
    text = re.sub(r"\*\*([A-Za-z]{1,4})\*\*([A-Za-z]{2,})", r"\1\2", text)
    for pat, repl in PHRASE_FIXES:
        text = re.sub(pat, repl, text, flags=re.I | re.M)
    def repl(m: re.Match) -> str:
        word = m.group(0)
        fix = OCR_WORD_FIXES.get(word.lower())
        if not fix:
            return word
        if word.isupper():
            return fix.upper()
        if word[0].isupper():
            return fix[:1].upper() + fix[1:]
        if fix[:1].isupper():
            return fix
        return fix.lower()

    return re.sub(r"\b[A-Za-z']+\b", repl, text)


def _cased(src: str, dest: str) -> str:
    if src.isupper():
        return dest.upper()
    if src[0].isupper():
        return dest[:1].upper() + dest[1:]
    return dest.lower()


def _osa_le(a: str, b: str, limit: int = 2) -> int:
    """Damerau (optimal string alignment) distance, bailing out above limit."""
    la, lb = len(a), len(b)
    if abs(la - lb) > limit:
        return limit + 1
    if a == b:
        return 0
    prev = list(range(lb + 1))
    prev2 = prev[:]
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            best = prev[j - 1] + cost
            ins = cur[j - 1] + 1
            if ins < best:
                best = ins
            delete = prev[j] + 1
            if delete < best:
                best = delete
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                transp = prev2[j - 2] + 1
                if transp < best:
                    best = transp
            cur[j] = best
            if best < row_min:
                row_min = best
        if row_min > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[lb] if prev[lb] <= limit else limit + 1


def _collect_edit1(word: str, known: set[str], lex: set[str]) -> tuple[set[str], set[str]]:
    """Same candidate set as Norvig edits1, without allocating tens of thousands of strings."""
    rank0: set[str] = set()
    rank1: set[str] = set()
    letters = "abcdefghijklmnopqrstuvwxyz"

    def consider(cand: str) -> None:
        if cand in known:
            rank0.add(cand)
        elif cand in lex:
            rank1.add(cand)

    for i, ch in enumerate(word):
        consider(word[:i] + word[i + 1 :])
        if i + 1 < len(word):
            consider(word[:i] + word[i + 1] + ch + word[i + 2 :])
    for a, b in CONFUSIONS:
        pos = word.find(a)
        if pos != -1:
            consider(word[:pos] + b + word[pos + len(a) :])
    for i in range(len(word)):
        prefix, rest = word[:i], word[i + 1 :]
        for c in letters:
            consider(prefix + c + rest)
    for i in range(len(word) + 1):
        prefix, rest = word[:i], word[i:]
        for c in letters:
            consider(prefix + c + rest)
    return rank0, rank1


def _looks_inflected(word: str, lex: set[str]) -> bool:
    """Mac's word list often omits regular plurals / -ed / -ing — treat those as valid."""
    if word.endswith("ies") and (word[:-3] + "y") in lex:
        return True
    if word.endswith(("ses", "ches", "shes", "xes", "zes")) and word[:-2] in lex:
        return True
    if word.endswith("es") and word[:-2] in lex:
        return True
    if word.endswith("s") and not word.endswith("ss") and word[:-1] in lex:
        return True
    if word.endswith("ed") and (word[:-2] in lex or word[:-1] in lex):
        return True
    if word.endswith("ing") and (word[:-3] in lex or (word[:-3] + "e") in lex):
        return True
    return False


def _is_known(word: str, lex: set[str], known: set[str]) -> bool:
    return word in lex or word in known or _looks_inflected(word, lex)


def _is_mathy(line: str) -> bool:
    if "$" in line or line.strip().startswith("\\"):
        return True
    sym = len(re.findall(r"[=+\-*/^_{}\\]", line))
    return sym >= 3 and sym * 2 >= max(1, len(re.findall(r"[A-Za-z]+", line)))


def page_vocab(text: str, extra: list[str]) -> set[str]:
    lex = lexicon()
    found = {w.lower() for w in extra if w.isalpha() and len(w) >= 3}
    for w in re.findall(r"[A-Za-z][A-Za-z']{2,}", text):
        lw = w.lower()
        if lw in lex:
            found.add(lw)
    found |= {w.lower() for w in SCHOOL_VOCAB}
    return found


def spellfix_text(text: str, hints: list[str]) -> str:
    """Replace likely OCR typos using the local dictionary + words seen on the page."""
    text = apply_ocr_word_fixes(text)
    lex = lexicon()
    known = page_vocab(text, hints)
    out_lines: list[str] = []
    for line in text.splitlines():
        fence = re.match(r"^(?:>\s*)+", line)
        prefix = fence.group(0) if fence else ""
        body = line[len(prefix):]
        if body.lstrip().startswith("[!") or _is_mathy(body):
            out_lines.append(line)
            continue

        def repl(m: re.Match) -> str:
            word = m.group(0)
            low = word.lower()
            if len(low) < 4 or not low.isalpha():
                return word
            if _is_known(low, lex, known):
                return word
            rank0, rank1 = _collect_edit1(low, known, lex)
            if rank0:
                return _cased(word, min(rank0))
            if rank1:
                if len(rank1) > 1:
                    return word
                return _cased(word, next(iter(rank1)))
            dist2 = [e for e in known if len(e) >= 4 and abs(len(e) - len(low)) <= 2 and _osa_le(low, e, 2) == 2]
            if not dist2:
                return word
            return _cased(word, min(dist2))

        out_lines.append(prefix + re.sub(r"\b[A-Za-z']+\b", repl, body))
    return "\n".join(out_lines)


def heading_md(title: str, level: int = 3) -> str:
    hashes = "#" * max(2, min(level, 4))
    clean = title.strip().rstrip(":").strip()
    clean = re.sub(r"^[*_#\s]+", "", clean)
    clean = re.sub(r"[*_]+$", "", clean).strip()
    return f"{hashes} <u>**{clean}**</u>"


def already_heading(line: str) -> bool:
    return bool(re.match(r"^#{1,6}\s+", line.strip()))


_ROMAN = re.compile(r"^[ivxlcdm]+$", re.I)
_SKIP_HEADING = {
    "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
    "yes", "no", "ok", "or", "and", "the", "a", "an",
}
_SMALL_WORDS = {"of", "the", "a", "an", "and", "or", "in", "on", "to", "for", "vs", "by"}
_KEY_IDEAS = {
    "the age of enlightenment", "the enlightenment", "scientific revolution",
    "natural rights", "natural law", "roots of democracy",
}


def heading_plain(line: str) -> str:
    s = re.sub(r"</?u>", "", line.strip(), flags=re.I)
    s = re.sub(r"^#{1,6}\s*", "", s)
    return re.sub(r"[*_`]+", "", s).strip()


def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or already_heading(s) or s.startswith("|") or s.startswith(">"):
        return False
    if s.startswith("- ") or s.startswith("* ") or re.match(r"^\d+[.)]\s", s):
        return False
    if "$" in s or s.startswith("\\") or s.startswith("<"):
        return False
    plain = re.sub(r"[*_`]+", "", s).strip()
    if len(plain) > 72 or len(plain) < 4:
        return False
    if re.search(r"\d", plain):
        return False
    if plain.endswith(".") and not plain.endswith("..."):
        return False
    words = plain.split()
    if len(words) > 10:
        return False
    if _ROMAN.match(plain) or plain.lower() in _SKIP_HEADING:
        return False
    letters = re.sub(r"[^A-Za-z]", "", plain)
    if len(letters) < 4:
        return False
    if re.sub(r"[*_]+", "", plain).lower().rstrip(":") in _KEY_IDEAS:
        return True
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    # Poster-style ALL CAPS only. Title Case body lines must stay body —
    # promoting them to headings chops a page into scattered titles.
    if upper_ratio >= 0.82 and 2 <= len(words) <= 8:
        return True
    if re.match(r"^[A-Z][A-Za-z0-9'’\-]*(?:\s+[A-Z][A-Za-z0-9'’\-]*){0,6}\s*:\s*$", plain):
        return True
    return False


def join_wrapped_lines(text: str) -> str:
    """Rejoin mid-sentence wraps, then keep real paragraph breaks."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        s = line.strip()
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        ps = prev.strip()
        if not s or not ps:
            out.append(line)
            continue
        if ps.startswith(("#", ">", "|", "-", "*")) or s.startswith(("#", ">", "|", "-", "*", "<")):
            out.append(line)
            continue
        if already_heading(ps) or already_heading(s) or _is_mathy(ps) or _is_mathy(s):
            out.append(line)
            continue
        if ps.endswith("?") or s.endswith("?"):
            out.append(line)
            continue
        if ps[-1] in ".!":
            out.append(line)
            continue
        if s[0].islower() or s[0] in ",;)" or ps.endswith("-"):
            joiner = "" if ps.endswith("-") else " "
            out[-1] = ps.rstrip("-") + joiner + s
            continue
        out.append(line)
    return "\n".join(out)


def looks_like_quote(line: str) -> bool:
    s = line.strip()
    if len(s) < 24 or s.startswith(">"):
        return False
    return s[0] in "\"“'" and s.count(" ") >= 4


def looks_like_directions(line: str) -> bool:
    s = line.strip()
    if len(s) < 48 or already_heading(s) or s.startswith(">"):
        return False
    low = s.lower()
    cues = (
        "we are going to", "you will", "you are going to",
        "this is a secondary", "practice annotation",
        "directions:", "instructions:", "in this activity",
    )
    return any(c in low for c in cues)


def format_stop_here(line: str) -> str | None:
    s = line.strip()
    m = re.match(r"^(?:STOP\s*HERE\s*:?\s*)(.+)$", s, flags=re.I)
    if m:
        return f"> **STOP HERE:** {m.group(1).strip()}"
    if re.match(r"^STOP\s*HERE\s*:?\s*$", s, flags=re.I):
        return "> **STOP HERE**"
    return None


def format_layout(text: str) -> str:
    def normalize_heading(m: re.Match) -> str:
        level = len(m.group(1))
        title = re.sub(r"</?u>", "", m.group(2), flags=re.I)
        title = re.sub(r"\*+", "", title).strip()
        if _ROMAN.match(title) or title.lower() in _SKIP_HEADING:
            return title
        return heading_md(title, level)

    text = html_cols_to_callouts(normalize_boxes(text))
    text = re.sub(
        r"^(#{2,4})\s+(?:<u>)?\*?\*?(.+?)\*?\*?(?:</u>)?\s*$",
        normalize_heading,
        text,
        flags=re.M,
    )
    text = re.sub(
        r"(?m)^((?:Key Idea|Main Idea|Big Idea|Essential Question):.*)$",
        r"**\1**",
        text,
    )
    text = re.sub(r"(?m)^Questions:\s*$", heading_md("Questions", 3), text)
    text = re.sub(r"(?m)^Questions:\s+(.+)$", lambda m: f"{heading_md('Questions', 3)}\n\n{m.group(1)}", text)

    lines_out: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            lines_out.append("")
            continue
        if is_model_leak(stripped):
            continue
        if stripped.startswith(">"):
            lines_out.append(line)
            continue
        stop = format_stop_here(stripped)
        if stop:
            lines_out.append(stop)
            continue
        if looks_like_quote(stripped):
            lines_out.append("> " + stripped)
            continue
        if looks_like_directions(stripped) and not stripped.startswith("*"):
            lines_out.append(f"*{stripped}*")
            continue
        if already_heading(stripped):
            if _ROMAN.match(heading_plain(stripped)):
                lines_out.append(heading_plain(stripped))
            else:
                lines_out.append(stripped)
            continue
        if looks_like_heading(stripped):
            plain = re.sub(r"[*_`]+", "", stripped).strip().rstrip(":").strip()
            level = 4 if (plain.isupper() or len(plain.split()) <= 4) else 3
            lines_out.append(heading_md(plain, level))
            continue
        if stripped.endswith("?") and 12 <= len(stripped) <= 140 and stripped[0].isupper():
            lines_out.append(f"**{stripped}**")
            continue
        title_body = re.match(
            r"^([A-Z][^.]{1,40}?)\s+[-—]\s+([A-Z][^.]{1,40}?)\s+[-—]\s+(.{20,})$",
            stripped,
        )
        if title_body and not stripped.startswith("http"):
            lines_out.append(heading_md(f"{title_body.group(1).strip()} — {title_body.group(2).strip()}", 3))
            lines_out.append("")
            lines_out.append(title_body.group(3).strip())
            continue
        defn = re.match(
            r"^([A-Z][A-Za-z0-9'’\-]*(?:\s+[A-Z][A-Za-z0-9'’\-]*){0,5})\s*([-—:])\s+(.+)$",
            stripped,
        )
        if defn and len(defn.group(1)) <= 48 and not stripped.startswith("http"):
            term, sep, rest = defn.group(1), defn.group(2), defn.group(3)
            if sep in "-—":
                lines_out.append(f"**{term}** — {rest}")
            else:
                lines_out.append(f"**{term}:** {rest}")
            continue
        lines_out.append(line)

    deduped: list[str] = []
    prev_norm = ""
    for ln in lines_out:
        n = norm_line(re.sub(r"[#*_<>/u]", "", ln))
        if n and already_heading(ln) and n == prev_norm:
            continue
        deduped.append(ln)
        prev_norm = n if ln.strip() else prev_norm
    text = "\n".join(deduped)
    text = re.sub(r"(?:\|(?:\s|---)+\|)+\n?", "", text)
    return space_out(join_wrapped_lines(text))


def space_out(text: str) -> str:
    """Air out headings, callouts, and paragraphs so the note is easier to read."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        prev = out[-1].strip() if out else ""
        if s.startswith(">"):
            if out and prev != "":
                out.append("")
            while i < len(lines):
                cur = lines[i]
                cur_s = cur.strip()
                nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if cur_s.startswith(">"):
                    out.append(cur.rstrip())
                    i += 1
                    continue
                if cur_s == "" and nxt.startswith(">"):
                    # A new callout must start outside this quote, or bands glue together.
                    if re.match(r"^>\s*\[!", nxt):
                        i += 1
                        break
                    out.append(">")
                    i += 1
                    continue
                break
            if out and out[-1].strip() != "":
                out.append("")
            continue
        is_question = (
            s.endswith("?")
            or (s.startswith("**") and s.endswith("**") and "?" in s)
        )
        if already_heading(s) or is_question:
            if out and prev != "":
                out.append("")
            out.append(line.rstrip())
            out.append("")
            if already_heading(s):
                out.append("")
            i += 1
            continue
        if (
            s
            and prev
            and not prev.startswith(("#", "-", "*", ">", "|"))
            and not s.startswith(("#", "-", "*", ">", "|"))
            and not _is_mathy(s)
            and not _is_mathy(prev)
            and (
                prev[-1:] in ".?!"
                or s.endswith("?")
                or (len(prev) > 60 and len(s) > 35)
            )
        ):
            out.append("")
        out.append(line.rstrip())
        i += 1
    text = "\n".join(out)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    return text


POLISH_PROMPT = """You clean OCR from a student worksheet. Return ONLY the Markdown body.

Hard rules:
- Fix misspelled / broken handwriting words using the sentence and the word list below.
- If a word is chopped (hole punch, torn edge, missing letters), complete it only when the sentence and the word list make the intended word obvious.
- Keep the FULL draft. Do not delete sentences, columns, or headings. If unsure, keep the draft wording.
- Do NOT invent examples or extra sections. Do NOT repeat a section.
- Keep ### <u>**Title**</u> headings, <u>underlines</u>, **bold**, > [!columns] / > [!columns3] / > [!left] / > [!mid] / > [!right], > [!box] frames, lists, and math.
- Leave a blank (non-quoted) line between column bands so they stay separate.
- Never emit HTML (<div>, style=, <span>).
- Always leave a blank line between paragraphs and around headings.
- Spell influence / influenced / Enlightenment correctly when the sentence needs those words.
- STOP HERE → > **STOP HERE:** …

Words also seen on this page (local Apple Vision; use them to choose spellings):
{hints}

OCR draft:
"""


def _heading_titles(text: str) -> list[str]:
    titles = []
    for ln in text.splitlines():
        if already_heading(ln):
            titles.append(norm_line(heading_plain(ln)))
    return [t for t in titles if t]


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{3,}", text))


def polish_acceptable(draft: str, polished: str) -> bool:
    if len(polished) < max(40, int(len(draft) * 0.8)):
        return False
    if _word_count(polished) < max(8, int(_word_count(draft) * 0.8)):
        return False
    if len(polished) > int(len(draft) * 1.55):
        return False
    dh, ph = _heading_titles(draft), _heading_titles(polished)
    if ph and len(ph) > max(12, len(dh) * 2 + 4):
        return False
    if ph and len(ph) != len(set(ph)) and len(set(ph)) < len(ph) * 0.6:
        return False
    if "[!columns]" in draft and "[!columns]" not in polished and "[!columns3]" not in polished:
        return False
    if "[!columns3]" in draft and "[!columns3]" not in polished:
        return False
    if "[!box]" in draft and "[!box]" not in polished:
        return False
    if re.search(r"<div\b|<span style=", polished, flags=re.I):
        return False
    draft_u = len(re.findall(r"</u>", draft, flags=re.I))
    polish_u = len(re.findall(r"</u>", polished, flags=re.I))
    if draft_u >= 2 and polish_u < max(1, draft_u // 2):
        return False
    if draft.count("\n\n") >= 3 and polished.count("\n\n") < max(1, draft.count("\n\n") // 3):
        return False
    return True


def finish_page(text: str, hints: list[str] | None = None, underlined: list[str] | None = None) -> str:
    """House style for every future page: spelling, underlines, key ideas, paragraph gaps."""
    text = html_cols_to_callouts(normalize_boxes(text))
    text = strip_instruction_leaks(text)
    text = spellfix_text(text, hints or [])
    text = join_wrapped_lines(text)
    text = apply_underlines(text, underlined or [])
    text = format_layout(text)
    text = separate_callouts(text)
    return text.strip() + "\n"


def separate_callouts(text: str) -> str:
    """Keep each notebook fold as its own callout — do not nest bands."""
    return re.sub(r"(?m)^>\s*\n(?=>\s*\[!columns)", "\n", text)


def polish_with_llm(text: str, model: str, hints: list[str]) -> str:
    progress(f"polish pass with {model}…")
    ollama_unload(ocr_model())
    hint_str = ", ".join(sorted({h for h in hints if h.isalpha() and len(h) >= 3})[:80]) or "(none)"
    n_pred = min(6144, max(2048, len(text.split()) * 3 + 512))
    out, _meta = ollama_generate(
        model=model,
        prompt=POLISH_PROMPT.format(hints=hint_str) + text,
        timeout=300,
        keep_alive="8m",
        num_predict=n_pred,
    )
    out = out.strip()
    if out.startswith("```"):
        out = re.sub(r"^```(?:markdown|md)?\s*", "", out)
        out = re.sub(r"\s*```$", "", out).strip()
    if not polish_acceptable(text, out):
        progress_done("polish skipped (kept spell-fixed draft — closer to the page)")
        return text
    progress_done(f"polished with {model}")
    return out


def load_hint_data(work: Path) -> dict:
    path = work / "vision_hints.json"
    empty = {"words": [], "underlined": [], "lines": []}
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return empty
    words = [str(w) for w in (data.get("words") or [])]
    for line in data.get("lines") or []:
        words.extend(re.findall(r"[A-Za-z][A-Za-z']{2,}", str(line)))
    underlined = [str(u) for u in (data.get("underlined") or []) if str(u).strip()]
    return {"words": words, "underlined": underlined, "lines": list(data.get("lines") or [])}


def load_hints(work: Path) -> list[str]:
    return load_hint_data(work)["words"]


_STOP_HINT = {
    "the", "and", "of", "to", "in", "is", "a", "for", "on", "with", "as", "by",
    "from", "that", "this", "are", "was", "be", "or", "an", "at", "it", "not",
}


def recover_missing_lines(body: str, vision_lines: list[str]) -> str:
    """Keep readable Vision lines that glm-ocr never transcribed.

    If glm-ocr mostly failed (prompt echo, stub), rebuild from Vision so the
    page is not emptied by leak-stripping.
    """
    lex = lexicon()
    vision_clean: list[str] = []
    for raw in vision_lines:
        line = apply_ocr_word_fixes(re.sub(r"\s+", " ", str(raw)).strip())
        if is_model_leak(line):
            continue
        words = re.findall(r"[A-Za-z']{3,}", line)
        if len(words) < 2:
            continue
        hits = sum(1 for w in words if w.lower() in lex or _looks_inflected(w.lower(), lex))
        if hits / max(1, len(words)) < 0.45:
            continue
        vision_clean.append(line)

    vis_words = _word_count(" ".join(vision_clean))
    body_words = _word_count(body)
    if vis_words >= 20 and body_words < max(18, int(vis_words * 0.28)):
        return "\n".join(vision_clean) + "\n"
    # Do not append leftover Vision fragments — that dumps lines at the bottom
    # out of page order and chops the note.
    return body


def content_tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z']{3,}", text.lower())
        if t not in _STOP_HINT
    }


def apply_underlines(text: str, phrases: list[str]) -> str:
    """Mark phrases Vision saw with a line under them on the hardcopy."""
    cleaned = [p.strip() for p in phrases if 6 <= len(p.strip()) <= 140]
    if len(cleaned) > 14:
        cleaned = [p for p in cleaned if len(p) <= 80]
    uniq = sorted(set(cleaned), key=len, reverse=True)
    for phrase in uniq:
        if re.search(r"[<>\\$|]", phrase):
            continue
        pattern = re.compile(rf"(?<!<u>)(?<!\w)({re.escape(phrase)})(?!\w)(?!</u>)", flags=re.I)

        def repl(m: re.Match) -> str:
            raw = m.group(1)
            start = m.start()
            window = m.string[max(0, start - 8): start]
            if "<u>" in window or raw.lstrip().startswith("#"):
                return raw
            return f"<u>{raw}</u>"

        text = pattern.sub(repl, text)
    return text


def looks_truncated(text: str, meta: dict) -> bool:
    if not (text or "").strip():
        return True
    reason = str(meta.get("done_reason") or "").lower()
    if reason == "length":
        return True
    if not meta.get("done", True):
        return True
    n_pred = int(meta.get("num_predict") or NUM_PREDICT)
    eval_count = int(meta.get("eval_count") or 0)
    return bool(n_pred and eval_count >= max(32, n_pred - 8))


def needs_continuation(text: str, meta: dict, vision_lines: list[str]) -> bool:
    if looks_truncated(text, meta):
        return True
    vis = _word_count(" ".join(str(x) for x in vision_lines))
    body = _word_count(text)
    return vis >= 80 and body < int(vis * 0.55)


def banner(n_tiles: int, columns: int) -> None:
    cprint("╭──────────────────────────────────────╮", color=MAGENTA)
    cprint("│  ocr-to-md  ·  local glm-ocr         │", color=MAGENTA)
    layout = {3: "three-column page", 2: "two-column page"}.get(columns, "full page")
    cprint(f"│  {n_tiles} pass(es)  ·  {layout:<18} │", color=MAGENTA)
    cprint(f"│  {ocr_model():<36} │", color=MAGENTA)
    cprint("╰──────────────────────────────────────╯", color=MAGENTA)


def _section_tuple(tile: dict, text: str) -> tuple:
    return (
        tile_kind(tile),
        tile_column(tile),
        text,
        tile_zone(tile),
        int(tile.get("y") or 0),
        int(tile.get("x") or 0),
        int(tile.get("h") or 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", nargs="?")
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument(
        "--finish-only",
        action="store_true",
        help="Apply house style to stdin Markdown (spacing, key ideas, spelling).",
    )
    args = parser.parse_args()
    if args.finish_only:
        sys.stdout.write(finish_page(sys.stdin.read()))
        return
    if not args.workdir:
        parser.error("workdir is required unless --finish-only")
    work = Path(args.workdir)
    manifest = json.loads((work / "manifest.json").read_text())
    hint_data = load_hint_data(work)
    hints = hint_data["words"]
    tiles = manifest.get("tiles") or []
    columns = int(manifest.get("columns") or 1)
    primaries = [t for t in tiles if not t.get("continuation")]
    continuations = [t for t in tiles if t.get("continuation")]
    if not primaries:
        primaries = tiles
        continuations = []
    banner(len(primaries), columns)

    sections: list[tuple] = []
    last_text = ""
    last_meta: dict = {}
    for i, tile in enumerate(primaries, 1):
        path = work / tile["file"]
        prompt = prompt_for(tile, columns)
        progress(f"reading {i}/{len(primaries)}  {tile['file']}  (full page)")
        text, meta = ollama_ocr(path, prompt)
        last_text, last_meta = text, meta
        sections.append(_section_tuple(tile, text))
        progress_done(f"done   {i}/{len(primaries)}  {tile['file']}")

    cprint("│ stitching + house style (local)…", color=DIM)
    body = stitch(sections)
    if continuations and needs_continuation(last_text, last_meta, hint_data.get("lines") or []):
        cprint("│ page ran long — reading lower half (still local)…", color=YELLOW)
        extra_sections: list[tuple] = []
        for tile in continuations:
            path = work / tile["file"]
            progress(f"reading lower  {tile['file']}")
            text, _meta = ollama_ocr(path, prompt_for(tile, columns))
            extra_sections.append(_section_tuple(tile, text))
            progress_done(f"done   lower  {tile['file']}")
        extra = stitch(extra_sections)
        if extra.strip():
            body = stitch_lines([body, extra])
    body = recover_missing_lines(body, hint_data.get("lines") or [])
    body = colorize(body, list(manifest.get("red_words") or []))
    body = finish_page(body, hints, hint_data["underlined"])

    if not args.no_polish and os.environ.get("OCR_NO_POLISH") != "1" and not LOW_MEM:
        model = pick_polish_model()
        if model:
            try:
                body = finish_page(
                    polish_with_llm(body, model, hints),
                    hints,
                    hint_data["underlined"],
                )
            except SystemExit as exc:
                cprint(f"│ polish failed ({exc}); keeping spell-fixed draft", color=YELLOW)
        else:
            cprint(
                "│ no polish model — local dictionary spelling only "
                f"(setup can install {POLISH_MODELS[0]})",
                color=YELLOW,
            )
    elif LOW_MEM or os.environ.get("OCR_NO_POLISH") == "1":
        cprint("│ skip LLM polish (8GB profile — glm-ocr stays loaded)", color=DIM)

    if not body.strip():
        raise SystemExit("glm-ocr returned no text")

    cprint("│ markdown ready", color=GREEN)
    sys.stdout.write(body)
    if not body.endswith("\n"):
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
