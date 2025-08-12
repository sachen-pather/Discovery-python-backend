# app.py - Enhanced with your friend's statistical improvements
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
import os
import pandas as pd
import numpy as np



# Import enhanced modules
try:
    from enhanced_debt_optimizer import get_enhanced_debt_optimization
    ENHANCED_DEBT_OPTIMIZER_AVAILABLE = True
except ImportError:
    # Fallback to original
    try:
        from debt_optimizer import get_debt_optimization
        ENHANCED_DEBT_OPTIMIZER_AVAILABLE = False
        print("⚠️ Using original debt optimizer")
    except ImportError:
        ENHANCED_DEBT_OPTIMIZER_AVAILABLE = False
        print("⚠️ Debt optimizer not available")

try:
    from enhanced_budget_analyzer import generate_enhanced_budget_report
    ENHANCED_BUDGET_ANALYZER_AVAILABLE = True
    print("✅ Enhanced budget analyzer available")
except ImportError:
    ENHANCED_BUDGET_ANALYZER_AVAILABLE = False
    print("⚠️ Enhanced budget analyzer not available")

try:
    from investment_analyzer import get_investment_analysis
    INVESTMENT_ANALYZER_AVAILABLE = True
except ImportError:
    INVESTMENT_ANALYZER_AVAILABLE = False
    print("⚠️ Investment analyzer not available")

app = Flask(__name__)

# Updated CORS configuration
CORS(app, 
     origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types."""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'dtype'):  # Other numpy types
        if 'int' in str(obj.dtype):
            return int(obj)
        elif 'float' in str(obj.dtype):
            return float(obj)
        else:
            return str(obj)
    elif pd is not None and pd.isna(obj):  # Safe pandas check
        return None
    else:
        return obj

def process_financial_data(csv_file_path):
    """Process CSV file and return financial analysis with enhanced features."""
    try:
        print(f"🔄 Starting to process: {csv_file_path}")
        print(f"📁 File exists: {os.path.exists(csv_file_path)}")
        
        if os.path.exists(csv_file_path):
            with open(csv_file_path, 'r') as f:
                first_lines = f.read(200)
                print(f"📄 First 200 chars of file: {first_lines}")
        
        # Use existing categorization logic
        print("🔄 Calling process_file...")
        categorization_result = process_file(csv_file_path)
        print(f"✅ process_file result: {categorization_result}")
        
        if not categorization_result:
            print("❌ process_file returned False")
            return None
        
        # Try enhanced budget report first
        categorized_file = os.path.join(OUTPUT_DIRECTORY, f"categorized_{os.path.basename(csv_file_path)}")
        print(f"📁 Looking for categorized file: {categorized_file}")
        print(f"📁 Categorized file exists: {os.path.exists(categorized_file)}")
        
        
        categorized_df = pd.read_csv(categorized_file)
        categorized_records = categorized_df.to_dict(orient="records")
        if ENHANCED_BUDGET_ANALYZER_AVAILABLE:
            print("🔄 Trying enhanced budget analysis...")
            try:
                enhanced_report = generate_enhanced_budget_report(categorized_file)
                print(f"✅ Enhanced report result: {enhanced_report is not None}")
                
                if enhanced_report:
                    print("✅ Using enhanced budget analysis")
                    result = {
                        "total_income": float(enhanced_report['analysis']['total_income']),
                        "total_expenses": float(enhanced_report['analysis']['total_expenses']),
                        "available_income": float(enhanced_report['analysis']['available_income']),
                        "category_breakdown": enhanced_report['analysis']['category_breakdown'],
                        "suggestions": enhanced_report['suggestions'],
                        "annuity_projection": enhanced_report['annuity_projection'],
                        "total_potential_savings": float(enhanced_report['total_potential_savings']),
                        "optimized_available_income": float(enhanced_report['optimized_available_income']),
                        "enhanced_mode": True,
                        "action_plan": enhanced_report.get('action_plan'),
                        "protected_categories": list(enhanced_report['analysis'].get('protected_categories_present', set())),
                        "transactions": categorized_records
                    }
                    print("✅ Successfully created enhanced result")
                    return convert_to_json_serializable(result)
            except Exception as enhanced_error:
                print(f"❌ Enhanced analysis failed: {enhanced_error}")
                import traceback
                traceback.print_exc()
        
        # Fallback to original budget report
        print("🔄 Falling back to original budget report...")
        try:
            report = generate_budget_report(categorized_file)
            print(f"✅ Original report result: {report is not None}")
            
            if not report:
                print("❌ Original report also failed")
                return None
            
            # Extract data for frontend
            analysis = report["analysis"]
            category_breakdown = analysis["category_breakdown"]
            total_expenses = analysis["total_expenses"]
            available_income = analysis["available_income"]
            
            # Calculate potential savings
            suggestions = generate_cost_cutting_suggestions(category_breakdown, total_expenses)
            total_potential_savings = sum(s.get('potential_savings', 0) for s in suggestions.values() if isinstance(s, dict))
            
            result = {
                "total_income": float(analysis["total_income"]),
                "total_expenses": float(total_expenses),
                "available_income": float(available_income),
                "category_breakdown": category_breakdown,
                "suggestions": suggestions,
                "annuity_projection": calculate_savings_annuity(available_income),
                "total_potential_savings": float(total_potential_savings),
                "optimized_available_income": float(available_income + total_potential_savings),
                "enhanced_mode": False
            }
            
            print("✅ Successfully created original result")
            return convert_to_json_serializable(result)
            
        except Exception as original_error:
            print(f"❌ Original analysis also failed: {original_error}")
            import traceback
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f"❌ Error processing financial data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
@app.route("/upload-csv", methods=["POST", "OPTIONS"])
def upload_csv():
    """Handle CSV file uploads with enhanced processing."""
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
    """Analyze debt payoff strategies with enhanced optimization."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    if not ENHANCED_DEBT_OPTIMIZER_AVAILABLE:
        return jsonify({"error": "Debt optimizer not available"}), 503
    
    try:
        data = request.get_json()
        available_monthly = data.get('available_monthly', 0)
        debts_csv_path = data.get('debts_csv_path')  # Optional custom path
        
        # Get enhanced debt optimization analysis
        if ENHANCED_DEBT_OPTIMIZER_AVAILABLE:
            result = get_enhanced_debt_optimization(available_monthly, debts_csv_path)
        else:
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
    """Get comprehensive financial analysis with enhanced features."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    try:
        data = request.get_json()
        
        # This endpoint expects that a budget analysis has already been run
        available_income = data.get('available_income', 0)
        optimized_available_income = data.get('optimized_available_income', available_income)
        debts_csv_path = data.get('debts_csv_path')
        enhanced_mode = data.get('enhanced_mode', False)
        
        comprehensive_result = {
            "budget_summary": {
                "available_income": available_income,
                "optimized_available_income": optimized_available_income,
                "potential_monthly_savings": optimized_available_income - available_income,
                "enhanced_mode": enhanced_mode
            }
        }
        
        # Add debt analysis if available and there's money for debt payments
        if ENHANCED_DEBT_OPTIMIZER_AVAILABLE and optimized_available_income > 0:
            try:
                if ENHANCED_DEBT_OPTIMIZER_AVAILABLE:
                    debt_result = get_enhanced_debt_optimization(optimized_available_income, debts_csv_path)
                else:
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
        
        # Enhanced recommendations based on available income
        recommendations = []
        
        if optimized_available_income <= 0:
            recommendations.append("Focus on expense reduction to free up money for debt payments and investments")
        elif optimized_available_income < 500:
            recommendations.append("Consider building an emergency fund first, then focus on high-interest debt")
            if enhanced_mode:
                recommendations.append("Enhanced analysis detected protected categories - focus on reducible expenses")
        elif optimized_available_income < 1500:
            recommendations.append("Split funds between debt payments and long-term investments")
            if enhanced_mode:
                recommendations.append("Use weighted optimization suggestions for maximum impact")
        else:
            recommendations.append("You have good capacity for both aggressive debt payoff and substantial investments")
            if enhanced_mode:
                recommendations.append("Enhanced analysis shows sophisticated optimization opportunities")
        
        comprehensive_result["recommendations"] = recommendations
        
        return jsonify(comprehensive_result), 200
        
    except Exception as e:
        return jsonify({"error": f"Comprehensive analysis failed: {str(e)}"}), 500

@app.route("/health", methods=["GET", "OPTIONS"])
def health_check():
    """Health check endpoint with enhanced feature status."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    return jsonify({
        "status": "healthy",
        "features": {
            "csv_upload": True,
            "pdf_upload": True,
            "ai_extraction": bool(OPENAI_API_KEY),
            "debt_optimizer": ENHANCED_DEBT_OPTIMIZER_AVAILABLE,
            "enhanced_debt_optimizer": ENHANCED_DEBT_OPTIMIZER_AVAILABLE,
            "enhanced_budget_analyzer": ENHANCED_BUDGET_ANALYZER_AVAILABLE,
            "investment_analyzer": INVESTMENT_ANALYZER_AVAILABLE,
            "protected_categories": ENHANCED_BUDGET_ANALYZER_AVAILABLE,
            "weighted_optimization": ENHANCED_BUDGET_ANALYZER_AVAILABLE
        }
    }), 200

@app.route("/supported-formats", methods=["GET", "OPTIONS"])
def supported_formats():
    """Return supported file formats with enhanced features."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    format_info = {
        "supported_formats": ["CSV", "PDF"],
        "csv_format": {
            "description": "Standard bank statement CSV",
            "required_columns": ["Date", "Description", "Amount (ZAR)", "Balance (ZAR)"],
            "optional_columns": ["ReduceAllowed"],
            "example": {
                "Date": "2025-07-01",
                "Description": "Salary – Acme Co",
                "Amount (ZAR)": "5600.0",
                "Balance (ZAR)": "5745.0",
                "ReduceAllowed": "true"
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
    }
    
    if ENHANCED_BUDGET_ANALYZER_AVAILABLE:
        format_info["enhanced_features"] = {
            "protected_categories": {
                "description": "Automatically detects and protects fixed obligations from optimization",
                "protected_keywords": ["loan", "repayment", "mortgage", "credit card", "monthly payment"],
                "manual_override": "Use ReduceAllowed column in CSV (true/false) to manually control"
            },
            "weighted_optimization": {
                "description": "Uses South African income brackets and spending patterns for realistic suggestions",
                "income_brackets": ["very_low", "low", "low_middle", "middle", "upper_middle"]
            }
        }
    
    return jsonify(format_info), 200

@app.route("/features", methods=["GET", "OPTIONS"])
def get_features():
    """Return available features with enhanced capabilities."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    features = {
        "budget_analysis": {
            "available": True,
            "description": "Categorize expenses and analyze spending patterns",
            "endpoints": ["/upload-csv", "/upload-pdf"],
            "enhanced": ENHANCED_BUDGET_ANALYZER_AVAILABLE
        },
        "debt_optimization": {
            "available": ENHANCED_DEBT_OPTIMIZER_AVAILABLE,
            "description": "Optimize debt payoff using avalanche or snowball strategies",
            "endpoints": ["/debt-analysis"],
            "requirements": ["debts.csv file with debt information"],
            "enhanced": ENHANCED_DEBT_OPTIMIZER_AVAILABLE
        },
        "investment_analysis": {
            "available": INVESTMENT_ANALYZER_AVAILABLE,
            "description": "Project investment returns for conservative, moderate, and aggressive portfolios",
            "endpoints": ["/investment-analysis"]
        },
        "comprehensive_analysis": {
            "available": ENHANCED_DEBT_OPTIMIZER_AVAILABLE or INVESTMENT_ANALYZER_AVAILABLE,
            "description": "Combined analysis of budget optimization, debt payoff, and investment projections",
            "endpoints": ["/comprehensive-analysis"],
            "enhanced": ENHANCED_BUDGET_ANALYZER_AVAILABLE and ENHANCED_DEBT_OPTIMIZER_AVAILABLE
        }
    }
    
    if ENHANCED_BUDGET_ANALYZER_AVAILABLE:
        features["protected_categories"] = {
            "available": True,
            "description": "Automatically identifies and protects fixed obligations from budget optimization",
            "features": [
                "Keyword-based detection of debt payments",
                "Category-based protection",
                "Manual override via CSV column",
                "Realistic optimization within constraints"
            ]
        }
        
        features["weighted_optimization"] = {
            "available": True,
            "description": "Uses South African economic context for realistic budget suggestions",
            "features": [
                "Income bracket analysis",
                "Regional cost considerations",
                "Priority-based action plans",
                "Confidence levels for suggestions"
            ]
        }
    
    if ENHANCED_DEBT_OPTIMIZER_AVAILABLE:
        features["enhanced_debt_analysis"] = {
            "available": True,
            "description": "Advanced debt optimization with budget report integration",
            "features": [
                "Automatic budget extraction from reports",
                "Bank statement integration",
                "Current payment detection",
                "Interest compound modeling"
            ]
        }
    
    return jsonify(features), 200

@app.route("/enhanced-features", methods=["GET", "OPTIONS"])
def get_enhanced_features():
    """New endpoint to showcase enhanced statistical features."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    enhanced_features = {
        "statistical_improvements": {
            "weighted_analysis": {
                "enabled": ENHANCED_BUDGET_ANALYZER_AVAILABLE,
                "description": "Uses statistical modeling based on South African household spending patterns",
                "benefits": [
                    "More accurate savings estimates",
                    "Income-bracket specific recommendations",
                    "Realistic optimization constraints"
                ]
            },
            "protected_category_detection": {
                "enabled": ENHANCED_BUDGET_ANALYZER_AVAILABLE,
                "description": "Automatically identifies fixed obligations that shouldn't be reduced",
                "detection_methods": [
                    "Keyword pattern matching",
                    "Category classification",
                    "Manual CSV overrides"
                ]
            },
            "compound_interest_modeling": {
                "enabled": ENHANCED_DEBT_OPTIMIZER_AVAILABLE,
                "description": "Advanced debt payoff calculations with multiple compounding methods",
                "options": ["nominal", "effective", "daily"]
            },
            "budget_report_integration": {
                "enabled": ENHANCED_DEBT_OPTIMIZER_AVAILABLE,
                "description": "Automatically extracts optimized savings from budget reports",
                "features": [
                    "Pattern recognition for budget amounts",
                    "Multi-format support",
                    "Fallback mechanisms"
                ]
            }
        },
        "algorithmic_enhancements": {
            "priority_scoring": {
                "enabled": ENHANCED_BUDGET_ANALYZER_AVAILABLE,
                "description": "Ranks optimization suggestions by potential impact",
                "scale": "1-5 priority levels"
            },
            "confidence_levels": {
                "enabled": ENHANCED_BUDGET_ANALYZER_AVAILABLE,
                "description": "Provides confidence estimates for savings projections",
                "levels": ["Low", "Medium", "High"]
            },
            "action_plan_generation": {
                "enabled": ENHANCED_BUDGET_ANALYZER_AVAILABLE,
                "description": "Creates time-based implementation plans",
                "categories": ["immediate_actions", "short_term_goals", "long_term_goals"]
            }
        },
        "data_quality": {
            "enhanced_categorization": {
                "enabled": True,
                "description": "Improved transaction categorization with AI fallback",
                "features": [
                    "Rule-based classification first",
                    "AI-powered uncertainty resolution",
                    "Caching for performance",
                    "Batch processing optimization"
                ]
            },
            "robust_parsing": {
                "enabled": True,
                "description": "Enhanced CSV and PDF processing",
                "improvements": [
                    "Flexible column detection",
                    "Error recovery mechanisms",
                    "Multiple format support"
                ]
            }
        }
    }
    
    return jsonify(enhanced_features), 200

if __name__ == "__main__":
    print("🚀 Starting Enhanced Financial Analyzer API")
    print("=" * 50)
    print(f"✅ Enhanced Budget Analyzer: {'Available' if ENHANCED_BUDGET_ANALYZER_AVAILABLE else 'Not Available'}")
    print(f"✅ Enhanced Debt Optimizer: {'Available' if ENHANCED_DEBT_OPTIMIZER_AVAILABLE else 'Not Available'}")
    print(f"✅ Investment Analyzer: {'Available' if INVESTMENT_ANALYZER_AVAILABLE else 'Not Available'}")
    print("=" * 50)
    
    app.run(host="0.0.0.0", port=5000, debug=True)