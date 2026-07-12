# extract-movie-questions

A small Python utility that mines a large general-knowledge trivia PDF and keeps **only** the movie / TV–related questions, producing a clean CSV dataset of `Page #, Question #, Question Text, Answer` rows.

It reads the source PDF one page at a time with PyPDF2, sends each page's text to Google's Gemini model with a prompt that filters out everything except questions about movies, TV series, cast/crew, awards, fictional characters, plot elements, or filming locations, and writes the filtered result to a per-page CSV. The consolidated dataset feeds the sibling `eval-text2sql` repo, which uses these questions to evaluate the `fastapi-text2sql` natural-language-to-SQL API.

For agent / contributor conventions see [AGENTS.md](AGENTS.md).

## What it does

- Opens [`10.000vragen.pdf`](10.000vragen.pdf) and iterates over its pages (PyPDF2 text extraction).
- For each page, asks Gemini to return only cinema/TV-relevant Q&A as CSV — ignoring geography, science, or general history unless tied to a famous production.
- Sanitizes the model reply (strips markdown code fences and stray prose), parses it as CSV, and normalizes every row to four columns, stamping the source page number.
- Writes one file per page into [`data/`](data/) as `10.000vragen-page-NNN.csv` (always with a header row, even when a page yields zero movie questions).

The merged result of all per-page files is `10.000vragen.csv` at the repo root. (Merging is done outside this script; the script only writes the per-page files.)

## Pipeline

1. **Extract** — `reader.pages[i].extract_text()` pulls raw text from a single PDF page. Processing always starts at page 3 (pages 1–2 are front matter) and runs through `LAST_PAGE` (or the end of the document).
2. **Filter** — the page text (truncated to 20,000 chars) is embedded in a prompt and sent to Gemini, which returns CSV containing only movie/TV questions.
3. **Normalize** — `_parse_csv_rows` drops headers and malformed rows; `_normalize_rows_with_page` forces the `Page #, Question #, Question Text, Answer` shape.
4. **Write** — the rows are written to `data/10.000vragen-page-NNN.csv` (UTF-8).
5. **Throttle & resume** — a 2-second pause follows each page; already-existing page CSVs are skipped, so re-running resumes where it left off.

## Configuration

Configuration is split between a `.env` file and constants at the top of [`extract-movie-questions.py`](extract-movie-questions.py).

`.env`:

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | Yes | Google Generative AI API key. The script exits with a `RuntimeError` if it is missing. |
| `GEMINI_MODEL` | No | Pin a specific Gemini model. If unset, the script auto-selects an available preferred model (`gemini-1.5-flash-latest`, then `-pro-latest`, then `gemini-1.0-pro`), falling back to `gemini-1.5-flash-latest`. |

Script constants:

| Constant | Default | Purpose |
|----------|---------|---------|
| `PDF_FILE_PATH` | `10.000vragen.pdf` | Input PDF (relative to the script). |
| `OUTPUT_DIR` | `data` | Where per-page CSVs are written. |
| `FIRST_PAGE` | `1` | 1-based first page to process (effective floor is 3). |
| `LAST_PAGE` | `None` | 1-based last page; `None` means "to end of document". |

> The `.env` file is gitignored. Never commit your API key.

## Running locally

```bash
# 1. Install dependencies (Python 3)
pip install -r requirements.txt

# 2. Provide your API key
echo "GEMINI_API_KEY=your-key-here" > .env

# 3. Run
python extract-movie-questions.py
```

Output appears in `data/` as the run progresses, with a per-page log line (rows written, or "no rows extracted"). On completion the script prints which Gemini model was used.

### Stopping a long run

To halt gracefully after the current page, create the stop flag:

```bash
# Windows PowerShell
New-Item -ItemType File data\stop-process.txt
```

The script checks for `data/stop-process.txt` after each page and exits cleanly if it finds it.

### Resuming

Because each page is written to its own CSV and existing files are skipped, simply re-running the script resumes processing. To re-extract a specific page, delete its `data/10.000vragen-page-NNN.csv` first.

## Notes

- **Encoding**: the source PDF is Dutch-titled and the dataset contains non-ASCII names. All CSVs are written as UTF-8 — read them as UTF-8.
- **Free-tier friendly**: the fixed 2-second inter-page delay and 20,000-char input cap are tuned to avoid rate-limit and oversize-input failures on Gemini's free tier.
- **Requirements**: [`PyPDF2`](requirements.txt), `google-generativeai`, `python-dotenv`.
