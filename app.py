# app.py - Enhanced with debt/investment split functionality
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

# NEW: Store the latest categorized file path for debt analysis
LATEST_CATEGORIZED_FILE = None

# NEW: Global variable to store current debt analysis
CURRENT_DEBT_ANALYSIS = None

# NEW: Global variable to store current debt/investment split
CURRENT_SPLIT = None

def clear_debt_analysis():
    """Clear any existing debt analysis and split when a new financial statement is uploaded."""
    global CURRENT_DEBT_ANALYSIS, CURRENT_SPLIT
    CURRENT_DEBT_ANALYSIS = None
    CURRENT_SPLIT = None
    print("🧹 Cleared previous debt analysis and split")

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
    global LATEST_CATEGORIZED_FILE
    
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
        
        # Store the categorized file path for debt analysis
        categorized_file = os.path.join(OUTPUT_DIRECTORY, f"categorized_{os.path.basename(csv_file_path)}")
        if os.path.exists(categorized_file):
            LATEST_CATEGORIZED_FILE = categorized_file
            print(f"💾 Stored latest categorized file: {LATEST_CATEGORIZED_FILE}")
        
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
                        "transactions": categorized_records,
                        "categorized_file_path": categorized_file,
                        
                        # NEW: Debt/investment split recommendations
                        "debt_to_income_ratio": enhanced_report.get('debt_to_income_ratio', 0),
                        "total_debt_payments": enhanced_report.get('total_debt_payments', 0),
                        "debt_payments_detected": enhanced_report.get('debt_payments_detected', []),
                        "recommended_debt_ratio": enhanced_report.get('recommended_debt_ratio', 0.5),
                        "recommended_investment_ratio": enhanced_report.get('recommended_investment_ratio', 0.5),
                        "split_rationale": enhanced_report.get('split_rationale', ''),
                        "recommended_debt_budget": enhanced_report.get('recommended_debt_budget', 0),
                        "recommended_investment_budget": enhanced_report.get('recommended_investment_budget', 0),
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
                "enhanced_mode": False,
                "transactions": categorized_records,
                "categorized_file_path": categorized_file,
                
                # DEFAULT: Basic split recommendations when enhanced not available
                "debt_to_income_ratio": 0,
                "recommended_debt_ratio": 0.5,
                "recommended_investment_ratio": 0.5,
                "split_rationale": "Basic 50/50 split - upload debt information for personalized recommendations",
                "recommended_debt_budget": (available_income + total_potential_savings) * 0.5,
                "recommended_investment_budget": (available_income + total_potential_savings) * 0.5,
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

    # IMPORTANT: Clear any existing debt analysis when uploading new statement
    clear_debt_analysis()

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

    # IMPORTANT: Clear any existing debt analysis when uploading new statement
    clear_debt_analysis()

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

@app.route("/apply-debt-investment-split", methods=["POST", "OPTIONS"])
def apply_debt_investment_split():
    """Apply user's debt/investment allocation split - NEW ENDPOINT"""
    global LATEST_CATEGORIZED_FILE, CURRENT_SPLIT
    
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    data = request.get_json()
    
    # Extract parameters
    total_available = data.get('total_available_income', 0)
    debt_ratio = data.get('debt_ratio', 0.5)  # 0.0 to 1.0
    investment_ratio = data.get('investment_ratio', 0.5)  # 0.0 to 1.0
    
    # Validate ratios sum to 1.0
    if abs(debt_ratio + investment_ratio - 1.0) > 0.01:
        return jsonify({"error": "Debt and investment ratios must sum to 1.0"}), 400
    
    result = {}
    
    # Calculate debt optimization with allocated budget
    if debt_ratio > 0 and ENHANCED_DEBT_OPTIMIZER_AVAILABLE:
        try:
            debt_result = get_enhanced_debt_optimization(
                total_available_income=total_available,
                debt_allocation_ratio=debt_ratio,
                categorized_file_path=LATEST_CATEGORIZED_FILE
            )
            result['debt_analysis'] = debt_result
        except Exception as e:
            result['debt_analysis'] = {"error": str(e)}
    
    # Calculate investment projections with allocated budget  
    if investment_ratio > 0 and INVESTMENT_ANALYZER_AVAILABLE:
        try:
            investment_result = get_investment_analysis(
                total_available_income=total_available,
                investment_allocation_ratio=investment_ratio
            )
            result['investment_analysis'] = investment_result
        except Exception as e:
            result['investment_analysis'] = {"error": str(e)}
    
    # Store the current split globally
    CURRENT_SPLIT = {
        "total_available": total_available,
        "debt_ratio": debt_ratio,
        "investment_ratio": investment_ratio,
        "debt_budget": total_available * debt_ratio,
        "investment_budget": total_available * investment_ratio
    }
    
    result.update({
        "split_applied": True,
        "debt_budget": total_available * debt_ratio,
        "investment_budget": total_available * investment_ratio,
        "ratios": {"debt": debt_ratio, "investment": investment_ratio}
    })
    
    return jsonify(convert_to_json_serializable(result))

@app.route("/current-split", methods=["GET", "OPTIONS"])
def get_current_split():
    """Get the current debt/investment split if available - NEW ENDPOINT"""
    global CURRENT_SPLIT
    
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    if CURRENT_SPLIT is None:
        return jsonify({"error": "No split applied", "has_split": False}), 404
    
    return jsonify({"has_split": True, "split": CURRENT_SPLIT}), 200

@app.route("/upload-debt-csv", methods=["POST", "OPTIONS"])
def upload_debt_csv():
    """Handle debt CSV file uploads and perform debt analysis."""
    global LATEST_CATEGORIZED_FILE, CURRENT_DEBT_ANALYSIS
    
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    if "file" not in request.files:
        return jsonify({"error": "No debt file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not file.filename.endswith(".csv"):
        return jsonify({"error": "Invalid or no debt CSV file"}), 400

    # Get available monthly amount for debt payments
    available_monthly = request.form.get('available_monthly', 0)
    try:
        available_monthly = float(available_monthly)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid available_monthly amount"}), 400

    # Save uploaded debt CSV file
    debt_file_path = os.path.join(UPLOAD_DIR, f"debt_{file.filename}")
    file.save(debt_file_path)

    try:
        # Validate debt CSV format
        import pandas as pd
        debt_df = pd.read_csv(debt_file_path)
        
        # Check if file is empty or has no debts
        if debt_df.empty or len(debt_df) == 0:
            print("📋 Empty debt file uploaded - no debts found")
            result = {
                "debts_uploaded": [],
                "debt_summary": {
                    "total_debts": 0,
                    "total_balance": 0,
                    "total_min_payments": 0,
                    "average_apr": 0,
                    "debt_types": {}
                },
                "message": "No debts found in uploaded file",
                "budget_used": available_monthly,
                "categorized_file_used": LATEST_CATEGORIZED_FILE
            }
            
            # Store empty result
            CURRENT_DEBT_ANALYSIS = result
            
            return jsonify(convert_to_json_serializable(result)), 200
        
        # Check required columns
        required_columns = ['name', 'balance', 'apr', 'min_payment', 'kind']
        missing_columns = [col for col in required_columns if col not in debt_df.columns]
        
        if missing_columns:
            return jsonify({
                "error": f"Missing required columns: {', '.join(missing_columns)}",
                "required_columns": required_columns,
                "found_columns": list(debt_df.columns)
            }), 400
        
        # Validate data types and values
        validation_errors = []
        
        for index, row in debt_df.iterrows():
            try:
                balance = float(row['balance'])
                apr = float(row['apr'])
                min_payment = float(row['min_payment'])
                
                if balance < 0:
                    validation_errors.append(f"Row {index + 1}: Balance cannot be negative")
                if apr < 0 or apr > 1:
                    validation_errors.append(f"Row {index + 1}: APR should be between 0 and 1 (e.g., 0.22 for 22%)")
                if min_payment < 0:
                    validation_errors.append(f"Row {index + 1}: Minimum payment cannot be negative")
                    
            except ValueError as e:
                validation_errors.append(f"Row {index + 1}: Invalid numeric value - {str(e)}")
        
        if validation_errors:
            return jsonify({
                "error": "Data validation failed",
                "validation_errors": validation_errors[:10],  # Limit to first 10 errors
                "example_format": {
                    "name": "Credit Card",
                    "balance": "8500.00",
                    "apr": "0.22",
                    "min_payment": "200.00",
                    "kind": "credit_card"
                }
            }), 400
        
        print(f"✅ Debt CSV validated successfully: {len(debt_df)} debts found")
        
        # Perform debt analysis using the uploaded file - UPDATED to use new parameters
        if not ENHANCED_DEBT_OPTIMIZER_AVAILABLE:
            return jsonify({"error": "Debt optimizer not available"}), 503
        
        print(f"🔄 Starting debt analysis with R{available_monthly} additional monthly budget")
        print(f"📁 Using categorized file: {LATEST_CATEGORIZED_FILE}")
        
        # UPDATED: Use new function signature with allocation ratio of 1.0 (100% to debt)
        result = get_enhanced_debt_optimization(
            total_available_income=available_monthly,
            debt_allocation_ratio=1.0,  # 100% allocation to debt
            debts_csv_path=debt_file_path,
            categorized_file_path=LATEST_CATEGORIZED_FILE
        )
        
        # Add debt summary to result
        debt_summary = {
            "total_debts": len(debt_df),
            "total_balance": float(debt_df['balance'].sum()),
            "total_min_payments": float(debt_df['min_payment'].sum()),
            "average_apr": float(debt_df['apr'].mean()),
            "debt_types": debt_df['kind'].value_counts().to_dict()
        }
        
        result["debt_summary"] = debt_summary
        result["debts_uploaded"] = debt_df.to_dict('records')
        
        # IMPORTANT: Store the debt analysis globally
        CURRENT_DEBT_ANALYSIS = result
        
        print("✅ Debt analysis completed successfully")
        
        return jsonify(convert_to_json_serializable(result)), 200
        
    except Exception as e:
        print(f"❌ Debt analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Debt analysis failed: {str(e)}"}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(debt_file_path):
            os.remove(debt_file_path)

@app.route("/debt-analysis", methods=["POST", "OPTIONS"])
def debt_analysis():
    """Analyze debt payoff strategies with enhanced optimization - UPDATED for new parameters."""
    global LATEST_CATEGORIZED_FILE
    
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    if not ENHANCED_DEBT_OPTIMIZER_AVAILABLE:
        return jsonify({"error": "Debt optimizer not available"}), 503
    
    try:
        data = request.get_json()
        available_monthly = data.get('available_monthly', 0)
        debts_csv_path = data.get('debts_csv_path')  # Optional custom path
        
        # Check if default debt file exists if no custom path provided
        if not debts_csv_path:
            default_debt_file = os.path.join(DATA_DIRECTORY, "debts.csv")
            if not os.path.exists(default_debt_file):
                return jsonify({
                    "error": "No debt file found. Please upload a debt CSV file using /upload-debt-csv endpoint",
                    "suggestion": "Use POST /upload-debt-csv with your debt information",
                    "required_format": {
                        "columns": ["name", "balance", "apr", "min_payment", "kind"],
                        "example": {
                            "name": "Credit Card",
                            "balance": "8500.00", 
                            "apr": "0.22",
                            "min_payment": "200.00",
                            "kind": "credit_card"
                        }
                    }
                }), 400
        
        # UPDATED: Use new function signature with allocation ratio of 1.0 (100% to debt)
        result = get_enhanced_debt_optimization(
            total_available_income=available_monthly,
            debt_allocation_ratio=1.0,  # 100% allocation to debt
            debts_csv_path=debts_csv_path,
            categorized_file_path=LATEST_CATEGORIZED_FILE
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Debt analysis failed: {str(e)}"}), 500

@app.route("/current-debt-analysis", methods=["GET", "OPTIONS"])
def get_current_debt_analysis():
    """Get the current debt analysis if available."""
    global CURRENT_DEBT_ANALYSIS
    
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    if CURRENT_DEBT_ANALYSIS is None:
        return jsonify({"error": "No debt analysis available", "has_analysis": False}), 404
    
    return jsonify({"has_analysis": True, "analysis": CURRENT_DEBT_ANALYSIS}), 200

@app.route("/investment-analysis", methods=["POST", "OPTIONS"])
def investment_analysis():
    """Analyze investment projections - UPDATED for new parameters."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    if not INVESTMENT_ANALYZER_AVAILABLE:
        return jsonify({"error": "Investment analyzer not available"}), 503
    
    try:
        data = request.get_json()
        available_monthly = data.get('available_monthly', 0)
        
        # UPDATED: Use new function signature with allocation ratio of 1.0 (100% to investment)
        result = get_investment_analysis(
            total_available_income=available_monthly,
            investment_allocation_ratio=1.0  # 100% allocation to investment
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Investment analysis failed: {str(e)}"}), 500

@app.route("/comprehensive-analysis", methods=["POST", "OPTIONS"])
def comprehensive_analysis():
    """Get comprehensive financial analysis with enhanced features - UPDATED."""
    global LATEST_CATEGORIZED_FILE
    
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
                # UPDATED: Use new function signature
                debt_result = get_enhanced_debt_optimization(
                    total_available_income=optimized_available_income,
                    debt_allocation_ratio=1.0,  # 100% allocation to debt for comprehensive analysis
                    debts_csv_path=debts_csv_path,
                    categorized_file_path=LATEST_CATEGORIZED_FILE
                )
                comprehensive_result["debt_analysis"] = debt_result
            except Exception as e:
                comprehensive_result["debt_analysis"] = {"error": f"Debt analysis failed: {str(e)}"}
        
        # Add investment analysis if available
        if INVESTMENT_ANALYZER_AVAILABLE and optimized_available_income > 0:
            try:
                # UPDATED: Use new function signature
                investment_result = get_investment_analysis(
                    total_available_income=optimized_available_income,
                    investment_allocation_ratio=1.0  # 100% allocation to investment for comprehensive analysis
                )
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
            "debt_csv_upload": True,
            "ai_extraction": bool(OPENAI_API_KEY),
            "debt_optimizer": ENHANCED_DEBT_OPTIMIZER_AVAILABLE,
            "enhanced_debt_optimizer": ENHANCED_DEBT_OPTIMIZER_AVAILABLE,
            "enhanced_budget_analyzer": ENHANCED_BUDGET_ANALYZER_AVAILABLE,
            "investment_analyzer": INVESTMENT_ANALYZER_AVAILABLE,
            "protected_categories": ENHANCED_BUDGET_ANALYZER_AVAILABLE,
            "weighted_optimization": ENHANCED_BUDGET_ANALYZER_AVAILABLE,
            "current_payment_detection": ENHANCED_DEBT_OPTIMIZER_AVAILABLE,
            "debt_analysis_clearing": True,
            "debt_investment_split": True,  # NEW FEATURE
        },
        "endpoints": {
            "budget_analysis": ["/upload-csv", "/upload-pdf"],
            "debt_analysis": ["/upload-debt-csv", "/debt-analysis"],
            "investment_analysis": ["/investment-analysis"],
            "comprehensive": ["/comprehensive-analysis"],
            "debt_status": ["/current-debt-analysis"],
            "split_management": ["/apply-debt-investment-split", "/current-split"],  # NEW ENDPOINTS
        },
        "latest_categorized_file": LATEST_CATEGORIZED_FILE,
        "current_debt_analysis": CURRENT_DEBT_ANALYSIS is not None,
        "current_split": CURRENT_SPLIT is not None  # NEW DEBUG INFO
    }), 200

if __name__ == "__main__":
    print("🚀 Starting Enhanced Financial Analyzer API with Debt/Investment Split Support")
    print("=" * 70)
    print(f"✅ Enhanced Budget Analyzer: {'Available' if ENHANCED_BUDGET_ANALYZER_AVAILABLE else 'Not Available'}")
    print(f"✅ Enhanced Debt Optimizer: {'Available' if ENHANCED_DEBT_OPTIMIZER_AVAILABLE else 'Not Available'}")
    print(f"✅ Investment Analyzer: {'Available' if INVESTMENT_ANALYZER_AVAILABLE else 'Not Available'}")
    print(f"✅ Debt/Investment Split: Available")
    print("=" * 70)
    
    app.run(host="0.0.0.0", port=5000, debug=True)