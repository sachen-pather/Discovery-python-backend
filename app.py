# backend/app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from bank_categorizer import process_file
from budget_analyzer import (
    generate_budget_report,
    generate_cost_cutting_suggestions,
    calculate_savings_annuity
)
from config import DATA_DIRECTORY, OUTPUT_DIRECTORY, OPENAI_API_KEY
from openai import OpenAI

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

@app.route("/upload-csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not file.filename.endswith(".csv"):
        return jsonify({"error": "Invalid or no CSV file"}), 400

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(file_path)

    if not process_file(file_path):
        os.remove(file_path)
        return jsonify({"error": "Processing failed"}), 500

    categorized_file = os.path.join(OUTPUT_DIRECTORY, f"categorized_{file.filename}")
    report = generate_budget_report(categorized_file)
    os.remove(file_path)

    if not report:
        return jsonify({"error": "Report generation failed"}), 500

    # Additional data for detailed insights
    category_breakdown = report["analysis"]["category_breakdown"]
    total_expenses = report["analysis"]["total_expenses"]
    available_income = report["analysis"]["available_income"]

    return jsonify({
        "total_income": report["analysis"]["total_income"],
        "total_expenses": total_expenses,
        "available_income": available_income,
        "category_breakdown": category_breakdown,
        "suggestions": generate_cost_cutting_suggestions(category_breakdown, total_expenses),
        "annuity_projection": calculate_savings_annuity(available_income)
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
