import PyPDF2
import google.generativeai as genai
import csv
import time
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---
PDF_FILE_PATH = "10.000vragen-page003-012.pdf"
OUTPUT_CSV = "movie_database_questions_001.csv"

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


model = genai.GenerativeModel(_select_model_name())

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
    
    Format the output as a CSV with three columns: Question #, Question Text, Answer.
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
    output_csv_path = os.path.join(base_dir, OUTPUT_CSV)

    print(f"Writing output CSV to: {output_csv_path}")

    if not os.path.exists(pdf_path):
        print("PDF file not found!")
        return

    # Initialize CSV file with headers
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Question #", "Question Text", "Answer"])

    # Read PDF
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(reader.pages)
        start_page = max(int(FIRST_PAGE), 1)
        end_page = int(LAST_PAGE) if LAST_PAGE is not None else total_pages
        end_page = min(end_page, total_pages)

        if start_page > end_page:
            print(f"Invalid page range: FIRST_PAGE={FIRST_PAGE}, LAST_PAGE={LAST_PAGE} (document has {total_pages} pages)")
            return

        pages_to_process = end_page - start_page + 1
        print(f"Processing {pages_to_process} pages (pages {start_page}-{end_page} of {total_pages})...")

        for i in range(start_page - 1, end_page):
            print(f"Extracting Page {i+1}/{total_pages}...")
            page_text = reader.pages[i].extract_text()
            
            # Get filtered results from Gemini
            csv_chunk = extract_movie_questions(page_text)

            if csv_chunk:
                preview = csv_chunk if len(csv_chunk) <= 1000 else (csv_chunk[:1000] + "\n...[truncated]...")
                print("Model output:\n" + preview)
            
            # Append results to the grand CSV
            if csv_chunk:
                with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
                    f.write(csv_chunk + "\n")
            else:
                text_len = 0 if not page_text else len(str(page_text))
                print(f"No CSV extracted for page {i+1} (extracted_text_len={text_len}).")
            
            # Small delay to respect API rate limits (Free tier)
            time.sleep(2)

    print(f"Extraction complete! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
