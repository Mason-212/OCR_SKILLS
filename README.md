# OCR_SKILLS

Latest **ocr-to-md** Cursor skill: local photo → Markdown for Obsidian hardcopy notes.

This is the skill only. Do not put class notes in this repo.

- Runs on your Mac (Ollama `glm-ocr`, then Apple Vision, then Tesseract) — nothing is uploaded
- Watches an `OCR Inbox` in every Obsidian vault under `~/school`
- On 8GB Apple Silicon (Mac Mini M1): uses `glm-ocr:q8_0`, 10k context, a ~1792px **full page** (not column tiles), quantized KV cache, skips the extra polish model so Metal does not swap
- Handles HEIC, auto-rotates sideways phone photos, keeps tables / boxes / ink colors
- House style: bold key ideas, underlines from the paper, blank lines between paragraphs, hole-punch words finished from context. glm-ocr reads the whole page the way Cursor OCR does

## Install (Mac Mini or any Mac)

```bash
git clone git@github.com:Mason-212/OCR_SKILLS.git ~/dev/OCR_SKILLS
mkdir -p ~/.cursor/skills
ln -sfn ~/dev/OCR_SKILLS/ocr-to-md ~/.cursor/skills/ocr-to-md
chmod +x ~/.cursor/skills/ocr-to-md/scripts/ocr-to-md \
         ~/.cursor/skills/ocr-to-md/scripts/watch-inbox \
         ~/.cursor/skills/ocr-to-md/scripts/glm_ocr.py \
         ~/.cursor/skills/ocr-to-md/scripts/setup.sh
```

Needs [Ollama](https://ollama.com) (open the app). `setup.sh` pulls `glm-ocr` (or `glm-ocr:q8_0` on 8GB Macs). Also needs Xcode Command Line Tools (`xcode-select --install`). Optional fallback: `brew install tesseract`.

Create empty class vaults if this Mac should not hold your real notes:

```bash
mkdir -p \
  "$HOME/school/Math/A2PCH/.obsidian" \
  "$HOME/school/Science/Living Earth/.obsidian" \
  "$HOME/school/Social_Studies/World Geo/.obsidian" \
  "$HOME/school/English/English 9/.obsidian" \
  "$HOME/school/electives/Mandarin/Mandarin 1/.obsidian"
```

Then:

```bash
bash ~/.cursor/skills/ocr-to-md/scripts/setup.sh
```

Drop a photo into **OCR Inbox**, or run:

```
/ocr-to-md path/to/photo.heic
```

Copy finished `.md` + the matching photo back to your school Mac with `cp -n` or `rsync --ignore-existing` so existing notes are not overwritten.
