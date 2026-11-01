# AGENTS.md - Agent Guide for extract-movie-questions

## Canary

Start every reply with "Philippe,": no exceptions, including short replies and tool summaries.

Agent rules only. What the script does, pipeline, configuration (`.env` and constants), running, stopping and resuming: @README.md (canonical). Keep it aligned with any behaviour change.

---

## Role in Agent BBB

Builds the cinema/TV question dataset consumed by `text2sql-eval` (its own repository since 2026-10-12). No dependency on the database or ChromaDB: the only output contract is the CSV schema `Page #, Question #, Question Text, Answer`; do not change it without updating the consumer. Sibling roster: `%USERPROFILE%/Nestor/projets/t2s-backlog/topics/related-repositories.txt`.

## Traps

- The stop flag the code checks is `data/stop-process.txt`; the empty `stop-process.txt` at the repo root is a legacy file, not the flag.
- `10.000vragen.csv` (consolidated dataset) is not produced by the script; merging is done outside it.
- Page range is set by editing `FIRST_PAGE` / `LAST_PAGE`, not by CLI args.

## Code conventions

Plain idiomatic Python, single module, no classes, **no Hungarian notation** (unlike most sibling repos). Private helpers `_*`; public functions `extract_movie_questions(text)` and `main()`; module constants UPPER_SNAKE.

## Secrets and encoding

`.env` (`GEMINI_API_KEY`) is gitignored: keep the key out of git history and out of any committed CSV or log. All CSVs are UTF-8 (`newline=''`), with non-ASCII names; never assume ASCII. Not containerized: run locally only.

---

**Last Updated**: 2026-10-31
**Current Version**: 1.0.3

## Backlog (Nestor second-brain)

No dedicated backlog file yet; use the cross-repo dashboard `C:\Users\vaugo\Nestor\projets\t2s-backlog\index.md` (local PC path, not on the VPS or cloud agents).
