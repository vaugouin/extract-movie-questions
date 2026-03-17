import PyPDF2
import google.generativeai as genai
import csv
import time
import os
import io
from dotenv import load_dotenv

# --- CONFIGURATION ---
PDF_FILE_PATH = "10.000vragen.pdf"
OUTPUT_DIR = "data"

# 1-based, inclusive page range to process.
# Set LAST_PAGE to None to process through the end of the document.
FIRST_PAGE = 1
LAST_PAGE = None

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY. Set it in your .env file (GEMINI_API_KEY=...) or in the environment.")

genai.configure(api_key=API_KEY)

def _select_model_name():
    env_model = os.getenv("GEMINI_MODEL")
    if env_model:
        return env_model
    """
    """
    preferred = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-1.0-pro",
    ]

    try:
        models = list(genai.list_models())

        def supports_generate_content(m):
            methods = getattr(m, "supported_generation_methods", None) or []
            return "generateContent" in methods

        # Try preferred names first if available
        for candidate in preferred:
            full_name = f"models/{candidate}"
            for m in models:
                if m.name == full_name and supports_generate_content(m):
                    return candidate

        # Otherwise pick the first Gemini model that supports generateContent
        for m in models:
            if supports_generate_content(m) and m.name.startswith("models/"):
                return m.name.replace("models/", "", 1)
    except Exception:
        pass

    return "gemini-1.5-flash-latest"


SELECTED_MODEL_NAME = _select_model_name()
model = genai.GenerativeModel(SELECTED_MODEL_NAME)

def _sanitize_model_csv_text(csv_text: str) -> str:
    if not csv_text:
        return ""

    lines = str(csv_text).splitlines()
    cleaned = []
    in_fenced_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fenced_block = not in_fenced_block
            continue
        if not in_fenced_block and stripped.lower().startswith("please provide"):
            continue
        if stripped:
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def _parse_csv_rows(csv_text: str):
    csv_text = _sanitize_model_csv_text(csv_text)
    if not csv_text:
        return []

    rows = []
    try:
        reader = csv.reader(io.StringIO(csv_text))
        for row in reader:
            if not row:
                continue
            if len(row) == 3 and [c.strip() for c in row] == ["Question #", "Question Text", "Answer"]:
                continue
            if len(row) == 4 and [c.strip() for c in row] == ["Page #", "Question #", "Question Text", "Answer"]:
                continue
            if len(row) not in (3, 4):
                continue
            rows.append(row)
    except Exception:
        return []

    return rows


def _normalize_rows_with_page(rows, page_number: int):
    normalized = []
    for row in rows:
        if len(row) == 3:
            normalized.append([str(page_number), row[0], row[1], row[2]])
        elif len(row) == 4:
            normalized.append([str(page_number), row[1], row[2], row[3]])
    return normalized

def extract_movie_questions(text):
    """Submits page text to Gemini to filter movie-related content."""
    if not text or not str(text).strip():
        return ""

    # Keep prompts within reasonable bounds to reduce failures on very long pages.
    # (Gemini has context limits; oversize inputs can lead to empty/blocked outputs.)
    text = str(text)
    if len(text) > 20000:
        text = text[:20000]

    prompt = f"""
    Below is a list of general knowledge questions and answers. 
    Extract only the questions applicable to a movie database (topics: movies, TV series, 
    cast/crew, awards, fictional characters, plot elements, or filming locations).
    
    Ignore any questions about geography, science, or general history unless they 
    directly relate to a famous production.

    In each page, questions are structure on three columns: Question #, Question Text 
    that can be on several lines, Answer that can be also on several lines. 

    Format the output as a CSV with four columns: Page #, Question #, Question Text, Answer.
    You can include line breaks in the CSV output, but only if it is relevant, like the question or answer contains an unumbered list. 
    Do not include a header or any extra text.
    
    TEXT:
    {text}
    """
    try:
        response = model.generate_content(prompt)

        # Don't rely on response.text, because it raises if the response has no valid parts.
        if getattr(response, "candidates", None):
            candidate = response.candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                out = "".join(getattr(p, "text", "") or "" for p in parts).strip()
                return out

            finish_reason = getattr(candidate, "finish_reason", None)
            print(f"Gemini returned no text parts (finish_reason={finish_reason}).")
            return ""

        print("Gemini returned no candidates.")
        return ""
    except Exception as e:
        print(f"Error calling API: {e}")
        return ""

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, PDF_FILE_PATH)
    data_dir = os.path.join(base_dir, OUTPUT_DIR)
    os.makedirs(data_dir, exist_ok=True)

    stop_flag_path = os.path.join(data_dir, "stop-process.txt")

    if not os.path.exists(pdf_path):
        print("PDF file not found!")
        return

    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]

    # Read PDF
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(reader.pages)
        start_page = max(int(FIRST_PAGE), 1)
        start_page = max(start_page, 3)
        end_page = int(LAST_PAGE) if LAST_PAGE is not None else total_pages
        end_page = min(end_page, total_pages)

        if start_page > end_page:
            print(f"Invalid page range: FIRST_PAGE={FIRST_PAGE}, LAST_PAGE={LAST_PAGE} (document has {total_pages} pages)")
            return

        pages_to_process = end_page - start_page + 1
        print(f"Processing {pages_to_process} pages (pages {start_page}-{end_page} of {total_pages})...")

        for i in range(start_page - 1, end_page):
            page_number = i + 1
            out_name = f"{pdf_base}-page-{page_number:03d}.csv"
            out_path = os.path.join(data_dir, out_name)

            if os.path.exists(out_path):
                print(f"Skipping page {page_number} (already exists): {out_name}")
                continue

            print(f"Extracting Page {page_number}/{total_pages}...")
            page_text = reader.pages[i].extract_text()
            
            # Get filtered results from Gemini
            csv_chunk = extract_movie_questions(page_text)
            rows = _normalize_rows_with_page(_parse_csv_rows(csv_chunk), page_number)

            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Page #", "Question #", "Question Text", "Answer"])
                if rows:
                    writer.writerows(rows)

            if rows:
                print(f"Wrote {len(rows)} rows to {os.path.join(OUTPUT_DIR, out_name)}")
            else:
                text_len = 0 if not page_text else len(str(page_text))
                print(f"No rows extracted for page {page_number} (extracted_text_len={text_len}).")
            
            # Small delay to respect API rate limits (Free tier)
            time.sleep(2)

            if os.path.exists(stop_flag_path):
                print(f"Stop flag detected at {os.path.join(OUTPUT_DIR, 'stop-process.txt')}. Exiting gracefully.")
                break

    print(f"Extraction complete! Gemini model used for inference: {SELECTED_MODEL_NAME}")

if __name__ == "__main__":
    main()
