# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class AnalysisRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    startup = db.Column(db.String(120))
    industry = db.Column(db.String(120))
    analysis_type = db.Column(db.String(50))  # "memo" or "market"
    result_path = db.Column(db.String(300))
    json_data = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class InvestmentMemo(db.Model):
    __tablename__ = "investment_memos"
    id = db.Column(db.Integer, primary_key=True)
    startup_name = db.Column(db.String(150), nullable=False)
    industry = db.Column(db.String(100), nullable=False)
    memo_html = db.Column(db.Text, nullable=False)
    pdf_path = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ComparableCompany(db.Model):
    __tablename__ = "comparable_companies"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    stage = db.Column(db.String(50))
    hq = db.Column(db.String(100))
    industry = db.Column(db.String(100))
    valuation_usd = db.Column(db.Float)
    revenue_usd = db.Column(db.Float)
    capital_raised_usd = db.Column(db.Float)
    burn_rate_usd_month = db.Column(db.Float)
    runway_months = db.Column(db.Float)
    churn_pct = db.Column(db.Float)
    retention_pct = db.Column(db.Float)
    cac_usd = db.Column(db.Float)
    ltv_usd = db.Column(db.Float)
    avg_deal_size_usd = db.Column(db.Float)
    tam_usd = db.Column(db.Float)
    sam_usd = db.Column(db.Float)
    som_usd = db.Column(db.Float)
    associated_startup = db.Column(db.String(150))  # link back to analyzed startup
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class IndustryBenchmark(db.Model):
    __tablename__ = "industry_benchmarks"
    id = db.Column(db.Integer, primary_key=True)
    industry = db.Column(db.String(100))
    region = db.Column(db.String(100))
    avg_burn_usd_month = db.Column(db.Float)
    avg_cac_usd = db.Column(db.Float)
    avg_ltv_usd = db.Column(db.Float)
    avg_runway_months = db.Column(db.Float)
    avg_churn_pct = db.Column(db.Float)
    avg_retention_pct = db.Column(db.Float)
    avg_tam_usd = db.Column(db.Float)
    avg_sam_usd = db.Column(db.Float)
    avg_som_usd = db.Column(db.Float)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
