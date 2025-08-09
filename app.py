# app.py - Updated with debt optimization and investment analysis
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

# Import new modules
try:
    from debt_optimizer import get_debt_optimization
    DEBT_OPTIMIZER_AVAILABLE = True
except ImportError:
    DEBT_OPTIMIZER_AVAILABLE = False
    print("⚠️ Debt optimizer not available")

try:
    from investment_analyzer import get_investment_analysis
    INVESTMENT_ANALYZER_AVAILABLE = True
except ImportError:
    INVESTMENT_ANALYZER_AVAILABLE = False
    print("⚠️ Investment analyzer not available")

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
        
        # Extract data for frontend
        analysis = report["analysis"]
        category_breakdown = analysis["category_breakdown"]
        total_expenses = analysis["total_expenses"]
        available_income = analysis["available_income"]
        
        # Calculate potential savings (use enhanced if available)
        if report.get('enhanced_suggestions'):
            suggestions = report['enhanced_suggestions']
            total_potential_savings = report['total_potential_savings']
        else:
            suggestions = generate_cost_cutting_suggestions(category_breakdown, total_expenses)
            total_potential_savings = sum(s.get('potential_savings', 0) for s in suggestions.values() if isinstance(s, dict))
        
        return {
            "total_income": analysis["total_income"],
            "total_expenses": total_expenses,
            "available_income": available_income,
            "category_breakdown": category_breakdown,
            "suggestions": suggestions,
            "annuity_projection": calculate_savings_annuity(available_income),
            "total_potential_savings": total_potential_savings,
            "optimized_available_income": available_income + total_potential_savings
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

@app.route("/debt-analysis", methods=["POST", "OPTIONS"])
def debt_analysis():
    """Analyze debt payoff strategies."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    if not DEBT_OPTIMIZER_AVAILABLE:
        return jsonify({"error": "Debt optimizer not available"}), 503
    
    try:
        data = request.get_json()
        available_monthly = data.get('available_monthly', 0)
        debts_csv_path = data.get('debts_csv_path')  # Optional custom path
        
        # Get debt optimization analysis
        result = get_debt_optimization(available_monthly, debts_csv_path)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Debt analysis failed: {str(e)}"}), 500

@app.route("/investment-analysis", methods=["POST", "OPTIONS"])
def investment_analysis():
    """Analyze investment projections."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    if not INVESTMENT_ANALYZER_AVAILABLE:
        return jsonify({"error": "Investment analyzer not available"}), 503
    
    try:
        data = request.get_json()
        available_monthly = data.get('available_monthly', 0)
        
        # Get investment analysis
        result = get_investment_analysis(available_monthly)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Investment analysis failed: {str(e)}"}), 500

@app.route("/comprehensive-analysis", methods=["POST", "OPTIONS"])
def comprehensive_analysis():
    """Get comprehensive financial analysis including budget, debt, and investment projections."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    try:
        data = request.get_json()
        
        # This endpoint expects that a budget analysis has already been run
        # and we're getting the optimized available income
        available_income = data.get('available_income', 0)
        optimized_available_income = data.get('optimized_available_income', available_income)
        debts_csv_path = data.get('debts_csv_path')
        
        comprehensive_result = {
            "budget_summary": {
                "available_income": available_income,
                "optimized_available_income": optimized_available_income,
                "potential_monthly_savings": optimized_available_income - available_income
            }
        }
        
        # Add debt analysis if available and there's money for debt payments
        if DEBT_OPTIMIZER_AVAILABLE and optimized_available_income > 0:
            try:
                debt_result = get_debt_optimization(optimized_available_income, debts_csv_path)
                comprehensive_result["debt_analysis"] = debt_result
            except Exception as e:
                comprehensive_result["debt_analysis"] = {"error": f"Debt analysis failed: {str(e)}"}
        
        # Add investment analysis if available
        if INVESTMENT_ANALYZER_AVAILABLE and optimized_available_income > 0:
            try:
                investment_result = get_investment_analysis(optimized_available_income)
                comprehensive_result["investment_analysis"] = investment_result
            except Exception as e:
                comprehensive_result["investment_analysis"] = {"error": f"Investment analysis failed: {str(e)}"}
        
        # Add recommendations based on available income
        recommendations = []
        
        if optimized_available_income <= 0:
            recommendations.append("Focus on expense reduction to free up money for debt payments and investments")
        elif optimized_available_income < 500:
            recommendations.append("Consider building an emergency fund first, then focus on high-interest debt")
        elif optimized_available_income < 1500:
            recommendations.append("Split funds between debt payments and long-term investments")
        else:
            recommendations.append("You have good capacity for both aggressive debt payoff and substantial investments")
        
        comprehensive_result["recommendations"] = recommendations
        
        return jsonify(comprehensive_result), 200
        
    except Exception as e:
        return jsonify({"error": f"Comprehensive analysis failed: {str(e)}"}), 500

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
            "ai_extraction": bool(OPENAI_API_KEY),
            "debt_optimizer": DEBT_OPTIMIZER_AVAILABLE,
            "investment_analyzer": INVESTMENT_ANALYZER_AVAILABLE
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
        },
        "debt_csv_format": {
            "description": "Debt information for optimization analysis",
            "required_columns": ["name", "balance", "apr", "min_payment", "kind"],
            "example": {
                "name": "Credit Card",
                "balance": "8500.00",
                "apr": "0.22",
                "min_payment": "200.00",
                "kind": "credit_card"
            },
            "note": "Place as 'debts.csv' in the same directory as the application"
        }
    }), 200

@app.route("/features", methods=["GET", "OPTIONS"])
def get_features():
    """Return available features and their status."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    return jsonify({
        "budget_analysis": {
            "available": True,
            "description": "Categorize expenses and analyze spending patterns",
            "endpoints": ["/upload-csv", "/upload-pdf"]
        },
        "debt_optimization": {
            "available": DEBT_OPTIMIZER_AVAILABLE,
            "description": "Optimize debt payoff using avalanche or snowball strategies",
            "endpoints": ["/debt-analysis"],
            "requirements": ["debts.csv file with debt information"]
        },
        "investment_analysis": {
            "available": INVESTMENT_ANALYZER_AVAILABLE,
            "description": "Project investment returns for conservative, moderate, and aggressive portfolios",
            "endpoints": ["/investment-analysis"]
        },
        "comprehensive_analysis": {
            "available": DEBT_OPTIMIZER_AVAILABLE or INVESTMENT_ANALYZER_AVAILABLE,
            "description": "Combined analysis of budget optimization, debt payoff, and investment projections",
            "endpoints": ["/comprehensive-analysis"]
        }
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)