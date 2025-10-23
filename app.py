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
    """
    Generate market analysis with comparable companies and
    aggregated industry-level insights by region.
    """
    if not deck_text.strip():
        raise ValueError("No readable text could be extracted from the deck")

    prompt = f"""
You are a venture capital analyst.
Analyze the following startup pitch deck and identify 5 comparable companies
at different funding stages (Pre-Seed, Seed, Series A, etc.).

For each comparable, return structured data with:
[
  {{
    "company": "Example Startup",
    "stage": "Seed",
    "hq": "US",
    "industry": "FinTech",
    "revenue_usd": 2500000,
    "valuation_usd": 18000000,
    "capital_raised_usd": 6000000,
    "burn_rate_usd_month": 180000,
    "runway_months": 12,
    "churn_pct": 8,
    "retention_pct": 92,
    "cac_usd": 1200,
    "ltv_usd": 15000,
    "tam_usd": 5000000000,
    "sam_usd": 1000000000,
    "som_usd": 200000000
  }}
]

Also include optional "stage_benchmarks" with averages for each funding stage.

Return ONLY valid JSON — no text outside of JSON.

Pitch Deck:
\"\"\"{deck_text[:4000]}\"\"\"
""".strip()

    def clean_json_output(text):
        """Extract JSON block safely from the GPT response."""
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")
        json_str = match.group(0)
        json_str = json_str.replace("'", '"')
        return json_str

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = _extract_response_text(response) or ""

        if DEBUG_AI_OUTPUT:
            with open(os.path.join(app.config["OUTPUT_FOLDER"], "logs", "market_analysis.txt"), "w", encoding="utf-8") as f:
                f.write(text)

        # Parse JSON
        json_str = clean_json_output(text)
        parsed = json.loads(json_str)

        comparables = parsed.get("comparables") if isinstance(parsed, dict) else parsed
        stage_benchmarks = parsed.get("stage_benchmarks", {}) if isinstance(parsed, dict) else {}

        if not isinstance(comparables, list):
            raise ValueError("Expected a list of comparables in response")

        # --- Compute Local Industry-Level Aggregates ---
        industry_summary = {}
        for c in comparables:
            industry = c.get("industry", "Unknown")
            region = c.get("hq", "Unknown")

            key = (industry, region)
            if key not in industry_summary:
                industry_summary[key] = {
                    "industry": industry,
                    "region": region,
                    "burns": [],
                    "cacs": [],
                    "ltvs": [],
                    "runways": [],
                    "churns": [],
                    "retentions": [],
                    "tams": [],
                    "sams": [],
                    "soms": []
                }

            def safe_val(x):
                return float(x) if isinstance(x, (int, float)) else None

            entry = industry_summary[key]
            entry["burns"].append(safe_val(c.get("burn_rate_usd_month")))
            entry["cacs"].append(safe_val(c.get("cac_usd")))
            entry["ltvs"].append(safe_val(c.get("ltv_usd")))
            entry["runways"].append(safe_val(c.get("runway_months")))
            entry["churns"].append(safe_val(c.get("churn_pct")))
            entry["retentions"].append(safe_val(c.get("retention_pct")))
            entry["tams"].append(safe_val(c.get("tam_usd")))
            entry["sams"].append(safe_val(c.get("sam_usd")))
            entry["soms"].append(safe_val(c.get("som_usd")))

        # Convert lists into averages
        def avg(lst):
            lst = [x for x in lst if x is not None]
            return round(sum(lst) / len(lst), 1) if lst else None

        aggregated = {}
        for (industry, region), data in industry_summary.items():
            aggregated[f"{industry} ({region})"] = {
                "region": region,
                "avg_burn_usd_month": avg(data["burns"]),
                "avg_cac_usd": avg(data["cacs"]),
                "avg_ltv_usd": avg(data["ltvs"]),
                "avg_runway_months": avg(data["runways"]),
                "avg_churn_pct": avg(data["churns"]),
                "avg_retention_pct": avg(data["retentions"]),
                "avg_tam_usd": avg(data["tams"]),
                "avg_sam_usd": avg(data["sams"]),
                "avg_som_usd": avg(data["soms"]),
            }

        # Merge GPT's benchmarks (if any) into our aggregates
        merged_benchmarks = {**stage_benchmarks, **aggregated}

        return {"comparables": comparables, "stage_benchmarks": merged_benchmarks}

    except Exception as e:
        print(f"❌ Market analysis failed: {e}")
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

        # --- Basic file validation ---
        if not file or not file.filename:
            raise ValueError("Please upload a PDF file")

        if not file.filename.lower().endswith('.pdf'):
            raise ValueError("Only PDF files are allowed")

        # --- Save uploaded file ---
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # --- Extract and analyze ---
        deck_text = extract_text(filepath)
        analysis = generate_market_analysis(deck_text)

        # analysis now looks like: {"comparables": [...], "stage_benchmarks": {...}}
        comparables = analysis.get("comparables", [])
        benchmarks = analysis.get("stage_benchmarks", {})

        if not comparables:
            return render_template(
                "error.html",
                error="Could not generate market analysis. Please try again."
            )

        # --- Optional: Compute summary stats (averages) ---
        vals = [c.get("valuation_usd", 0) for c in comparables if isinstance(c.get("valuation_usd"), (int, float))]
        revs = [c.get("revenue_usd", 0) for c in comparables if isinstance(c.get("revenue_usd"), (int, float))]
        avg_val = round(sum(vals) / len(vals), 1) if vals else 0
        avg_rev = round(sum(revs) / len(revs), 1) if revs else 0

        # --- Persist the full analysis in database ---
        record = AnalysisRecord(
            startup="Uploaded Pitch Deck",
            industry="",
            analysis_type="market",
            result_path="",
            json_data=json.dumps(analysis)  # includes both comparables + benchmarks
        )
        db.session.add(record)
        db.session.commit()

        # --- Render results ---
        return render_template(
            "market_result.html",
            company="Uploaded Pitch Deck",
            comparables=comparables,
            stage_benchmarks=benchmarks,
            avg_valuation=avg_val,
            avg_revenue=avg_rev,
        )

    # --- GET request: show upload form ---
    return render_template("market_analyzer.html")


# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
