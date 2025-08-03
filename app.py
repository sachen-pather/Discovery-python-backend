# app.py - Fixed CORS configuration
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from bank_categorizer import process_file
from budget_analyzer import (
    generate_budget_report,
    generate_cost_cutting_suggestions,
    calculate_savings_annuity
)
from pdf_extractor import pdf_to_csv
from config import DATA_DIRECTORY, OUTPUT_DIRECTORY, OPENAI_API_KEY
from openai import OpenAI

app = Flask(__name__)

# Updated CORS configuration - Allow all origins and methods
CORS(app, 
     origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

def process_financial_data(csv_file_path):
    """Process CSV file and return financial analysis."""
    try:
        # Use existing categorization logic
        if not process_file(csv_file_path):
            return None
        
        # Generate budget report
        categorized_file = os.path.join(OUTPUT_DIRECTORY, f"categorized_{os.path.basename(csv_file_path)}")
        report = generate_budget_report(categorized_file)
        
        if not report:
            return None
        
        category_breakdown = report["analysis"]["category_breakdown"]
        total_expenses = report["analysis"]["total_expenses"]
        available_income = report["analysis"]["available_income"]
        
        return {
            "total_income": report["analysis"]["total_income"],
            "total_expenses": total_expenses,
            "available_income": available_income,
            "category_breakdown": category_breakdown,
            "suggestions": generate_cost_cutting_suggestions(category_breakdown, total_expenses),
            "annuity_projection": calculate_savings_annuity(available_income)
        }
        
    except Exception as e:
        print(f"❌ Error processing financial data: {e}")
        return None

@app.route("/upload-csv", methods=["POST", "OPTIONS"])
def upload_csv():
    """Handle CSV file uploads."""
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not file.filename.endswith(".csv"):
        return jsonify({"error": "Invalid or no CSV file"}), 400

    # Save uploaded CSV file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(file_path)

    try:
        # Process the CSV file
        result = process_financial_data(file_path)
        
        if not result:
            return jsonify({"error": "CSV processing failed"}), 500

        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"CSV processing failed: {str(e)}"}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route("/upload-pdf", methods=["POST", "OPTIONS"])
def upload_pdf():
    """Handle PDF file uploads - convert to CSV then process."""
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not file.filename.endswith(".pdf"):
        return jsonify({"error": "Invalid or no PDF file"}), 400

    # Save uploaded PDF file
    pdf_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(pdf_path)

    try:
        # Convert PDF to CSV
        csv_filename = file.filename.replace('.pdf', '_extracted.csv')
        csv_path = os.path.join(UPLOAD_DIR, csv_filename)
        
        print(f"🔄 Converting PDF to CSV: {file.filename}")
        
        if not pdf_to_csv(pdf_path, csv_path):
            return jsonify({"error": "Failed to extract data from PDF"}), 500
        
        print(f"✅ PDF converted to CSV: {csv_filename}")
        
        # Process the extracted CSV using existing logic
        result = process_financial_data(csv_path)
        
        if not result:
            return jsonify({"error": "Failed to process extracted data"}), 500

        return jsonify(result), 200
        
    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        return jsonify({"error": f"PDF processing failed: {str(e)}"}), 500
    finally:
        # Clean up temporary files
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if 'csv_path' in locals() and os.path.exists(csv_path):
            os.remove(csv_path)

@app.route("/health", methods=["GET", "OPTIONS"])
def health_check():
    """Health check endpoint."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    return jsonify({
        "status": "healthy",
        "features": {
            "csv_upload": True,
            "pdf_upload": True,
            "ai_extraction": bool(OPENAI_API_KEY)
        }
    }), 200

@app.route("/supported-formats", methods=["GET", "OPTIONS"])
def supported_formats():
    """Return supported file formats."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    return jsonify({
        "supported_formats": ["CSV", "PDF"],
        "csv_format": {
            "description": "Standard bank statement CSV",
            "required_columns": ["Date", "Description", "Amount (ZAR)", "Balance (ZAR)"],
            "example": {
                "Date": "2025-07-01",
                "Description": "Salary – Acme Co",
                "Amount (ZAR)": "5600.0",
                "Balance (ZAR)": "5745.0"
            }
        },
        "pdf_format": {
            "description": "Bank statement PDF (text-based, not scanned)",
            "note": "PDF will be converted to CSV format automatically"
        }
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)