from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv
import markdown
from weasyprint import HTML
import re
import json
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from functools import wraps
from werkzeug.utils import secure_filename
from models import db, AnalysisRecord


# --- Setup ---
load_dotenv()
DEBUG_AI_OUTPUT = os.getenv("DEBUG_AI_OUTPUT", "False").lower() == "true"
OCR_ENABLED = os.getenv("OCR_ENABLED", "True").lower() == "true"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)


app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///kb_intelligence.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    db.create_all()

print("🧭 Using template folder:", os.path.abspath(app.template_folder))
print("🧭 Using static folder:", os.path.abspath(app.static_folder))

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Folder setup
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(BASE_DIR, "outputs")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
os.makedirs(os.path.join(app.config["OUTPUT_FOLDER"], "logs"), exist_ok=True)

# OpenAI client
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    print("⚠️ OPENAI_API_KEY not set in environment")
client = OpenAI(api_key=_api_key)


# Helper to extract text from OpenAI response (handles SDK shapes)
def _extract_response_text(response):
    try:
        # object-like (new OpenAI client)
        return response.choices[0].message.content
    except Exception:
        try:
            # dict-like
            return response["choices"][0]["message"]["content"]
        except Exception:
            try:
                return response.choices[0].text
            except Exception:
                return None


# --- Helper: Extract text from PDF ---
def extract_text(file_path):
    """Extract text from PDF using PyMuPDF, fallback to OCR if needed."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    text = ""
    errors = []

    # Try PyMuPDF first
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text")
    except Exception as e:
        errors.append(f"PyMuPDF extraction failed: {e}")

    # Fallback to OCR if needed
    if not text.strip() and OCR_ENABLED:
        try:
            pages = convert_from_path(file_path, dpi=200)
            for page in pages:
                text += pytesseract.image_to_string(page)
        except Exception as e:
            errors.append(f"OCR extraction failed: {e}")

    if not text.strip():
        raise ValueError(f"Could not extract text from PDF. Errors: {'; '.join(errors)}")

    return text.strip()

# --- Helper: Generate investment memo ---
def generate_memo(startup_name, industry, deck_text):
    if not deck_text.strip():
        raise ValueError("No readable text could be extracted from the uploaded deck")

    prompt = f"""
You are an experienced venture capital analyst. Based on the following startup pitch deck, write a concise, structured investment memo.

Startup: {startup_name}
Industry: {industry}

Pitch Deck Text:
\"\"\"{deck_text[:4000]}\"\"\"

Write a memo with:
# Executive Summary
# Market Landscape
# Product Overview
# Competition
# Risks & Recommendation
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )

        content = _extract_response_text(response)
        if not content:
            raise ValueError("No text found in OpenAI response")

        if DEBUG_AI_OUTPUT:
            with open(os.path.join(app.config["OUTPUT_FOLDER"], "logs", "last_memo.txt"), "w", encoding="utf-8") as f:
                f.write(content)

        return content

    except Exception as e:
        print(f"❌ OpenAI memo generation failed: {e}")
        raise ValueError(f"Failed to generate memo: {str(e)}")

# --- Helper: Generate market analysis from uploaded deck ---
def generate_market_analysis(deck_text):
    """Generate market analysis with comparable companies from deck text."""
    if not deck_text.strip():
        raise ValueError("No readable text could be extracted from the deck")

    prompt = f"""
You are a venture capital analyst.
Analyze the following startup pitch deck and identify 5 comparable companies.
For each comparable, return numeric metrics as integers (no commas, no currency symbols).

Return ONLY valid JSON (nothing else).
Use this exact structure:
[
  {{
    "name": "Company A",
    "valuation_musd": 150,
    "arr_musd": 10,
    "funding_musd": 25,
    "hq_country": "US"
  }}
]

Pitch Deck:
\"\"\"{deck_text[:4000]}\"\"\"""".strip()

    def clean_json_output(text):
        """Clean and extract JSON from model output."""
        # Remove markdown fences and whitespace
        text = re.sub(r"```(?:json)?", "", text).strip()
        # Find JSON array
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in response")
        # Clean the JSON string
        json_str = match.group(0)
        json_str = re.sub(r'(?<=\d),(?=\d)', '', json_str)  # Remove thousands separators
        json_str = json_str.replace("'", '"')  # Fix quotes
        return json_str

    try:
        # First attempt
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = _extract_response_text(response) or ""
        
        # Log raw output if debug enabled
        if DEBUG_AI_OUTPUT:
            with open(os.path.join(app.config["OUTPUT_FOLDER"], "logs", "market_analysis.txt"), "w", encoding="utf-8") as f:
                f.write(text)

        # Parse and validate response
        json_str = clean_json_output(text)
        data = json.loads(json_str)
        
        # Validate structure
        if not isinstance(data, list):
            raise ValueError("Response is not a list of companies")
        if not data:
            raise ValueError("No comparable companies found")
        
        # Validate each company has required fields
        required_fields = {"name", "valuation_musd", "arr_musd", "funding_musd", "hq_country"}
        for company in data:
            if not isinstance(company, dict):
                raise ValueError("Each comparable must be an object")
            missing = required_fields - set(company.keys())
            if missing:
                raise ValueError(f"Company missing required fields: {missing}")

        return data

    except Exception as e:
        print(f"❌ Market analysis failed: {str(e)}")
        print("Raw output snippet:", (text[:200] if 'text' in locals() else "No output generated"))
        raise ValueError(f"Failed to generate market analysis: {str(e)}")

# Error handling decorator
def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print(f"❌ Error in {f.__name__}: {str(e)}")
            return render_template("error.html", 
                                 error=str(e),
                                 back_url=request.referrer or url_for('index'))
    return decorated_function


# ---------------- ROUTES ----------------

@app.route("/")
def index():
    return render_template("index.html")
    
@app.route("/dashboard")
def dashboard():
    records = AnalysisRecord.query.order_by(AnalysisRecord.created_at.desc()).all()
    return render_template("dashboard.html", records=records)

# -------- Investment Memo Generator --------
@app.route("/upload_memo", methods=["GET", "POST"])
@handle_errors
def upload_memo():
    if request.method == "POST":
        startup_name = request.form.get("startup_name")
        industry = request.form.get("industry")
        file = request.files.get("file")

        if not startup_name or not industry:
            raise ValueError("Please provide both startup name and industry")
        
        if not file or not file.filename:
            raise ValueError("Please upload a PDF file")

        if not file.filename.lower().endswith('.pdf'):
            raise ValueError("Only PDF files are allowed")

        # sanitize filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Extract & generate memo
        deck_text = extract_text(filepath)
        memo_markdown = generate_memo(startup_name, industry, deck_text)
        memo_html = markdown.markdown(memo_markdown)

        # Save as PDF
        safe_startup = secure_filename(startup_name) or "memo"
        output_filename = f"{safe_startup}_memo.pdf"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
        pdf_html = render_template(
            "memo_result.html",
            startup=startup_name,
            memo_html=memo_html,
            download_link=url_for("download_file", filename=output_filename),
        )
        try:
            HTML(string=pdf_html).write_pdf(output_path)
        except Exception as e:
            print("⚠️ PDF generation failed:", e)

        # Persist record (correct indentation)
        record = AnalysisRecord(
            startup=startup_name,
            industry=industry,
            analysis_type="memo",
            result_path=output_filename,
            json_data=memo_markdown
        )
        db.session.add(record)
        db.session.commit()

        return render_template(
            "memo_result.html",
            startup=startup_name,
            memo_html=memo_html,
            download_link=url_for("download_file", filename=output_filename),
        )

    return render_template("upload_memo.html")

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

# -------- Market Analyzer (PDF Upload) --------
@app.route("/market_analyzer", methods=["GET", "POST"])
@handle_errors
def market_analyzer():
    if request.method == "POST":
        file = request.files.get("file")

        if not file or not file.filename:
            raise ValueError("Please upload a PDF file")

        if not file.filename.lower().endswith('.pdf'):
            raise ValueError("Only PDF files are allowed")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Extract text & generate comparables
        deck_text = extract_text(filepath)
        comparables = generate_market_analysis(deck_text)
        
        if not comparables:
            return render_template(
                "error.html",
                error="Could not generate market analysis. Please try again."
            )
        
        # Compute averages
        vals = [c.get("valuation_musd", 0) for c in comparables if isinstance(c.get("valuation_musd"), (int, float))]
        arrs = [c.get("arr_musd", 0) for c in comparables if isinstance(c.get("arr_musd"), (int, float))]
        avg_val = round(sum(vals) / len(vals), 1) if vals else 0
        avg_arr = round(sum(arrs) / len(arrs), 1) if arrs else 0

        # Persist record (correct indentation)
        record = AnalysisRecord(
            startup="Uploaded Pitch Deck",
            industry="",
            analysis_type="market",
            result_path="",
            json_data=json.dumps(comparables)
        )
        db.session.add(record)
        db.session.commit()

        # Render dynamic result
        return render_template(
            "market_result.html",
            company="Uploaded Pitch Deck",
            comparables=[
                {
                    "name": c.get("name"),
                    "valuation": c.get("valuation_musd"),
                    "arr": c.get("arr_musd"),
                    "funding": c.get("funding_musd"),
                    "hq": c.get("hq_country"),
                }
                for c in comparables
            ],
            avg_valuation=avg_val,
            avg_arr=avg_arr,
        )

    return render_template("market_analyzer.html")

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
