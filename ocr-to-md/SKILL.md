---
name: ocr-to-md
description: >-
  Converts photos of hardcopy notes to Markdown using local OCR (Ollama glm-ocr,
  Apple Vision fallback). Use when the user says ocr-to-md, OCR, scan notes,
  convert an image/photo/screenshot/HEIC of handwritten or printed notes to
  markdown, or set up an Obsidian OCR Inbox watcher.
---

# OCR to Markdown (local)

Runs **on this Mac**. Does not send images to a cloud OCR API.

## Usage

```
/ocr-to-md path/to/photo.jpg
```

Redo a page after engine improvements:

```
/ocr-to-md --force path/to/photo.heic
```

Re-install the always-on inbox watcher (all vaults under `~/school`):

```
/ocr-to-md setup
```

See the inbox queue with `/ocr-check`. Kill a named page and send it to the back of the queue with `/ocr-terminate IMG_3389.heic`.

## What the agent must do

1. Resolve the image path(s) (absolute paths).
2. Run the bundled script — do **not** call a remote OCR tool.

```bash
~/.cursor/skills/ocr-to-md/scripts/ocr-to-md --force "/abs/path/to/image.jpg"
```

**Setup / repair the auto watcher:**

```bash
bash ~/.cursor/skills/ocr-to-md/scripts/setup.sh
```

3. Report the written `.md` path(s). Do not dump the full OCR text unless the user asks to review it.

A dense handwritten page takes **about 2–6 minutes** (one local glm-ocr pass on the whole page). On an 8GB M1 Mac Mini the watcher uses `glm-ocr:q8_0`, 10k context, a ~1792px full-page image, quantized KV cache, and no extra polish-model swap so Metal does not page. Quality first — it will not silently fall back to the fast Apple Vision engine if glm-ocr is installed.

## Automatic Obsidian flow

Each class vault under `~/school` has an **OCR Inbox** folder.

- Drop a photo into **OCR Inbox**, or paste it into any note.
- A sibling `.md` appears after glm-ocr finishes.
- Log: `~/Library/Logs/ocr-to-md.log`

## Pipeline (default `auto`)

1. Load the full-resolution image (HEIC included).
2. Auto-rotate so handwriting is upright (sideways phone photos).
3. Light shadow/contrast lift for handwriting (local Core Image — not heavy sharpen).
4. Detect layout with on-device Apple Vision (hints only: spelling, underlines, 1- vs 2-column). Do **not** crop the page into column/zone tiles.
5. Detect underlined phrases from the ink on the page (local pixel check).
6. Run local **Ollama `glm-ocr`** on the **whole page** (official `Text Recognition:` prompt), the same way Cursor OCR sees a photo. Finish words eaten by a hole punch / binder ring from the rest of the sentence; do not invent new facts. If generation hits the token cap, read the overlapping lower half once and merge.
7. House-style the single transcript (do not stitch left/right crops). Drawn boxes as `> [!box]` only when the model (or the page) marked them.
8. **House style on every page** (this is the default for all future scans): blank line between paragraphs; key ideas/titles like *The Age of Enlightenment* as `### <u>**Title**</u>`; hand underlines as `<u>…</u>`; questions bold; long quotes as blockquotes; spelling from the page + Mac dictionary (e.g. influence). Reading order comes from glm-ocr seeing the full page — notes are not chopped or shuffled by tiles.
9. Optional local `llama3.2:3b` polish (skipped on 8GB so glm-ocr stays loaded), then house style runs again so polish cannot strip spacing or underlines.
10. Write markdown with `![[image]]` above the transcript.

Fallback (only if glm-ocr is not installed): Apple Vision, then Tesseract — still run through the same house style.

## Output

```markdown
# Lecture notes

![[Lecture notes.heic]]

### <u>**Section title**</u>

Body text…

> [!columns]
>
> > [!left]
> > Left column from the page
>
> > [!right]
> > Right column from the page

> [!box] Answer box
> Text that sat inside a drawn frame

> **STOP HERE:** question from the worksheet
```

## Flags

| Flag | Meaning |
|------|---------|
| `--force` | Overwrite existing sibling `.md` |
| `--stdout` | Print markdown; do not write a file |
| `--engine auto\|glm\|vision\|tesseract` | Pick OCR engine |
