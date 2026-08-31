# AGENTS.md - Agent Guide for extract-movie-questions

This file gives you the agentic context you need to work on this codebase safely. For project overview, features, install / deploy steps and human-facing security / performance / troubleshooting material, read @README.md — that file is canonical and not duplicated here.

This is the single canonical guide for autonomous coding agents in this repository. Assistant-specific files such as @CLAUDE.md, and any future tool-specific guide such as `GEMINI.md`, should only point here and should not duplicate repository instructions.

- For any project update, keep documentation aligned:
  - Update `README.md` for user-facing behavior, configuration, setup, deployment, troubleshooting, or verification changes.
  - Update this file only when agent workflow or safety context changes.

---

## Related repositories (project ecosystem)

This repo is one stage of the larger **Agent BBB** movie/TV database pipeline (all siblings live under `github.com/vaugouin/<repo>`). It builds the curated cinema/TV **evaluation question dataset** — the `Page #, Question #, Question Text, Answer` CSV rows extracted here — that is consumed by the sibling `eval-text2sql` repo. `eval-text2sql` in turn uses those questions to score the `fastapi-text2sql` natural-language-to-SQL API against the shared MySQL/MariaDB database (`T_WC_*` tables). This repo has no direct dependency on the database or ChromaDB; its only output contract is the CSV schema.

The canonical sibling-repo roster lives in `%USERPROFILE%/Nestor/projets/t2s-backlog/topics/related-repositories.txt`.

## Where things live (file → role)

| Path | Role |
|------|------|
| `extract-movie-questions.py` | The entire program. Reads the PDF page-by-page, sends each page's text to Gemini, parses/normalizes the CSV reply, writes one CSV per page. |
| `10.000vragen.pdf` | Input source — a large general-knowledge trivia PDF (path hard-coded as `PDF_FILE_PATH`). |
| `data/` | Output directory (`OUTPUT_DIR`). One file per processed page: `10.000vragen-page-NNN.csv` (3-digit zero-padded). |
| `data/stop-process.txt` | Cooperative stop-flag. If present after a page completes, the run exits gracefully. (A legacy empty `stop-process.txt` at repo root is **not** the one the code checks.) |
| `10.000vragen.csv` | Consolidated dataset (all per-page rows merged). This is an output artifact; the script does **not** produce it — merging is done outside this script. |
| `requirements.txt` | `PyPDF2`, `google-generativeai`, `python-dotenv`. |
| `.env` | Holds `GEMINI_API_KEY` (and optional `GEMINI_MODEL`). Gitignored. |

## Code conventions

Plain idiomatic Python, single module, no classes. Helper functions are underscore-prefixed and private (`_select_model_name`, `_sanitize_model_csv_text`, `_parse_csv_rows`, `_normalize_rows_with_page`); the two public functions are `extract_movie_questions(text)` and `main()`. Module-level constants are UPPER_SNAKE (`PDF_FILE_PATH`, `OUTPUT_DIR`, `FIRST_PAGE`, `LAST_PAGE`). No Hungarian notation. Page range (`FIRST_PAGE` / `LAST_PAGE`) is configured by editing the constants, not via CLI args; `LAST_PAGE = None` means "to end of document".

## Configuration & secrets

- **`GEMINI_API_KEY`** (required) — loaded via `python-dotenv` `load_dotenv()`; the script raises `RuntimeError` at startup if missing.
- **`GEMINI_MODEL`** (optional) — pins the model. If unset, `_select_model_name()` queries `genai.list_models()` and picks the first available preferred model (`gemini-1.5-flash-latest` → `gemini-1.5-pro-latest` → `gemini-1.0-pro`), falling back to `gemini-1.5-flash-latest`.
- **Secrets handling**: `.env` is gitignored — never commit it. If you regenerate or rotate the key, keep it out of git history and out of any committed CSV/log.
- **Resumability**: per-page output makes runs idempotent — if `data/<base>-page-NNN.csv` already exists, that page is skipped. Delete a page's CSV to force re-extraction.
- **Rate limiting**: a hard-coded `time.sleep(2)` runs after each page to respect the Gemini free-tier limit. Oversized page text is truncated to 20000 chars before the prompt to avoid empty/blocked responses.
- **Hard page floor**: regardless of `FIRST_PAGE`, processing starts no earlier than page 3 (`start_page = max(start_page, 3)`) — pages 1–2 are front matter.
- **Stop flag**: create `data/stop-process.txt` to halt a long run cleanly after the current page.

## Encoding

The dataset is multilingual / contains non-ASCII names (the source PDF is Dutch-titled — "10.000 vragen"). All CSV output is written with `encoding='utf-8'` and `newline=''`. Always read and write these files as UTF-8; do not assume ASCII.

## Deployment

This repo is **not containerized**: there is no `Dockerfile`, `docker-compose.yml`, or any container deployment. It is run locally/ad hoc as a single Python script (`python extract-movie-questions.py`) against the hard-coded input PDF, after `pip install -r requirements.txt` and providing `.env`.

---

**Last Updated**: 2026-06-03
**Current Version**: 1.0.0

## Backlog (Nestor second-brain)

The implementation backlog for the Agent BBB ecosystem lives in the **Nestor** knowledge repo
(a separate repo, not cloned alongside this one). This repo has no dedicated backlog file yet;
use the cross-repo dashboard:

- Dashboard: `C:\Users\vaugo\Nestor\projets\t2s-backlog\index.md`

NOTE: this is a local path on Philippe's PC and does not resolve on the VPS or on cloud agents.
