#!/usr/bin/env bash
# Create OCR Inbox folders in every school vault and install the always-on watcher.
# Ensures local Ollama models (glm-ocr + small polish model) are available.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHOOL_ROOT="${SCHOOL_ROOT:-$HOME/school}"
PLIST_DEST="$HOME/Library/LaunchAgents/com.user.ocr-to-md.plist"
LOG_PATH="$HOME/Library/Logs/ocr-to-md.log"
WATCHER="$SKILL_DIR/scripts/watch-inbox"
POLISH_MODEL="${OCR_POLISH_MODEL:-llama3.2:3b}"

if [[ -t 1 ]]; then
  C_MAGENTA=$'\033[35m'
  C_CYAN=$'\033[36m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_BOLD=$'\033[1m'
  C_RESET=$'\033[0m'
else
  C_MAGENTA=""; C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_BOLD=""; C_RESET=""
fi

INBOX_NOTE='# OCR Inbox

Drop a photo of a worksheet or handwritten notes **in this folder**.

A matching `.md` file appears next to it after OCR finishes (often 2–6 minutes for a dense page — slower on purpose so it stays on glm-ocr). Everything runs **on this Mac** — nothing is uploaded.

Every future page gets the same house style: **bold** key ideas and titles, <u>underlines</u> where you underlined, a blank line between paragraphs, hole-punch words finished from context, and spelling fixed from the rest of the page. Columns stay split; drawn boxes stay boxes. All on this Mac.

You can also paste a photo into any note in this vault. Obsidian saves the image here, and the same markdown file is created.

Already have a photo somewhere else in the vault? In Cursor:

```
/ocr-to-md path/to/photo.jpg
```

Redo after engine updates:

```
/ocr-to-md --force path/to/photo.jpg
```
'

if [[ ! -d "$SCHOOL_ROOT" ]]; then
  echo "ERROR: school folder not found: $SCHOOL_ROOT"
  exit 1
fi

if [[ ! -x "$WATCHER" ]]; then
  echo "ERROR: watcher not executable: $WATCHER"
  exit 1
fi

printf '%s╭──────────────────────────────────────╮%s\n' "$C_MAGENTA" "$C_RESET"
printf '%s│%s  ocr-to-md setup  ·  all vaults      %s│%s\n' "$C_MAGENTA" "$C_RESET" "$C_MAGENTA" "$C_RESET"
printf '%s╰──────────────────────────────────────╯%s\n' "$C_MAGENTA" "$C_RESET"
printf '  school root : %s\n' "$SCHOOL_ROOT"
printf '  watcher     : %s\n' "$WATCHER"
printf '  polish model: %s\n\n' "$POLISH_MODEL"

# --- Local models (Ollama) -------------------------------------------------
ensure_ollama() {
  if ! command -v ollama >/dev/null 2>&1; then
    printf '%s│%s ollama not found — install from https://ollama.com (OCR will fall back to Vision)\n' "$C_YELLOW" "$C_RESET"
    return 1
  fi
  if ! curl -sf --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    printf '%s│%s starting Ollama…\n' "$C_CYAN" "$C_RESET"
    open -a Ollama 2>/dev/null || true
    for _ in $(seq 1 20); do
      curl -sf --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
      sleep 1
    done
  fi
  if ! curl -sf --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    printf '%s│%s Ollama API not reachable — skip model pull\n' "$C_YELLOW" "$C_RESET"
    return 1
  fi
  return 0
}

ensure_model() {
  local name="$1"
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$name"; then
    printf '%s│%s model ready: %s\n' "$C_GREEN" "$C_RESET" "$name"
    return 0
  fi
  # Also accept untagged family match (e.g. glm-ocr)
  if ollama list 2>/dev/null | awk '{print $1}' | grep -q "^${name%%:*}"; then
    printf '%s│%s model ready: %s (installed)\n' "$C_GREEN" "$C_RESET" "$name"
    return 0
  fi
  printf '%s│%s pulling %s (local only, one-time)…\n' "$C_CYAN" "$C_RESET" "$name"
  if ollama pull "$name"; then
    printf '%s│%s pulled %s\n' "$C_GREEN" "$C_RESET" "$name"
  else
    printf '%s│%s could not pull %s — continuing without it\n' "$C_YELLOW" "$C_RESET" "$name"
    return 1
  fi
}

if ensure_ollama; then
  ensure_model "glm-ocr" || true
  ensure_model "$POLISH_MODEL" || true
fi

echo ""

vault_count=0
while IFS= read -r -d '' obsidian; do
  vault="$(dirname "$obsidian")"
  inbox="$vault/OCR Inbox"
  mkdir -p "$inbox"
  printf '%s' "$INBOX_NOTE" > "$inbox/How OCR Inbox works.md"

  python3 - "$vault/.obsidian/app.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
data = {}
if p.exists() and p.read_text().strip():
    data = json.loads(p.read_text())
data["attachmentFolderPath"] = "OCR Inbox"
p.write_text(json.dumps(data, indent=2) + "\n")
PY

  snippet_dir="$vault/.obsidian/snippets"
  mkdir -p "$snippet_dir"
  cp "$SKILL_DIR/snippets/ocr-notes.css" "$snippet_dir/ocr-notes.css"
  python3 - "$vault/.obsidian/appearance.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
data = {}
if p.exists() and p.read_text().strip():
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        data = {}
if not isinstance(data, dict):
    data = {}
enabled = data.get("enabledCssSnippets") or []
if "ocr-notes" not in enabled:
    enabled.append("ocr-notes")
data["enabledCssSnippets"] = enabled
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(data, indent=2) + "\n")
PY

  printf '%s│%s inbox ready : %s\n' "$C_GREEN" "$C_RESET" "$inbox"
  vault_count=$((vault_count + 1))
done < <(find "$SCHOOL_ROOT" -type d -name '.obsidian' -print0)

if [[ "$vault_count" -eq 0 ]]; then
  echo "ERROR: no Obsidian vaults found under $SCHOOL_ROOT"
  exit 1
fi

printf '\n%s│%s compiling local OCR helpers…\n' "$C_CYAN" "$C_RESET"
compile_swift() {
  local src="$1" bin="$2"
  if [[ -x "$bin" && "$bin" -nt "$src" ]]; then
    printf '%s│%s ready: %s\n' "$C_GREEN" "$C_RESET" "$(basename "$bin")"
    return 0
  fi
  if xcrun swiftc -O -o "$bin" "$src"; then
    printf '%s│%s compiled %s\n' "$C_GREEN" "$C_RESET" "$(basename "$bin")"
  else
    printf '%s│%s could not compile %s — will run via swift\n' "$C_YELLOW" "$C_RESET" "$(basename "$src")"
  fi
}
compile_swift "$SKILL_DIR/scripts/prepare_page.swift" "$SKILL_DIR/scripts/prepare_page_bin"
compile_swift "$SKILL_DIR/scripts/ocr_to_md.swift" "$SKILL_DIR/scripts/ocr_to_md_bin"

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

cat > "$PLIST_DEST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.user.ocr-to-md</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$WATCHER</string>
    <string>--discover</string>
    <string>$SCHOOL_ROOT</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ProcessType</key>
  <string>Background</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>StandardOutPath</key>
  <string>$LOG_PATH</string>
  <key>StandardErrorPath</key>
  <string>$LOG_PATH</string>
</dict>
</plist>
EOF

uid="$(id -u)"
launchctl bootout "gui/$uid/com.user.ocr-to-md" 2>/dev/null || true
launchctl bootstrap "gui/$uid" "$PLIST_DEST"

printf '\n%s│%s Watching %s%s%s vault(s). Log: %s\n' \
  "$C_GREEN" "$C_RESET" "$C_BOLD" "$vault_count" "$C_RESET" "$LOG_PATH"
printf '%s│%s Reload Obsidian vaults so pasted images go to OCR Inbox.\n' "$C_CYAN" "$C_RESET"
printf '%s│%s Drop a photo in any OCR Inbox — or redo with /ocr-to-md --force\n' "$C_CYAN" "$C_RESET"
