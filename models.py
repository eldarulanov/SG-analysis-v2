# KB_Project_Two/models.py
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
