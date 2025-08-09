# budget_analyzer.py - Your original file with minimal additions for new features
import pandas as pd
import glob
import os
from datetime import datetime
import re
import math

# Import the enhanced optimizer (optional - graceful fallback if not available)
try:
    from enhanced_budget_analyzer import generate_enhanced_cost_cutting_suggestions
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

# Configuration
CATEGORIZED_DATA_DIR = os.path.join(os.path.dirname(__file__), "categorized_output")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "budget_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def find_amount_column(df):
    """Find the amount column with different possible names."""
    possible_names = [
        'Amount', 'amount', 'AMOUNT',
        'Amount (ZAR)', 'Amount(ZAR)', 'amount_zar',
        'Debit', 'debit', 'DEBIT',
        'Credit', 'credit', 'CREDIT',
        'Transaction Amount', 'transaction_amount',
        'Value', 'value', 'VALUE'
    ]
    
    for col in df.columns:
        if col in possible_names:
            return col
        # Check for partial matches
        col_lower = col.lower()
        if any(name.lower() in col_lower for name in ['amount', 'debit', 'credit', 'value']):
            return col
    
    return None

def find_description_column(df):
    """Find the description column."""
    possible_names = [
        'Description', 'description', 'DESCRIPTION',
        'Transaction Description', 'transaction_description',
        'Details', 'details', 'DETAILS',
        'Narration', 'narration', 'NARRATION'
    ]
    
    for col in df.columns:
        if col in possible_names:
            return col
    return None

def categorize_income_expense(description, amount):
    """Determine if a transaction is income or expense based on description and amount."""
    if pd.isna(description) or pd.isna(amount):
        return 'Other'
    
    desc_lower = str(description).lower()
    
    # Skip administrative entries like Opening Balance
    if any(word in desc_lower for word in ['opening balance', 'closing balance']):
        return 'Other'
    
    # Income indicators
    income_keywords = [
        'salary', 'wage', 'income', 'payment received', 'deposit', 
        'refund', 'cashback', 'interest', 'dividend', 'bonus',
        'transfer in', 'credit', 'reversal'
    ]
    
    # Check if it's likely income (positive amount + income keywords OR large positive amount)
    if amount > 0:
        if any(keyword in desc_lower for keyword in income_keywords):
            return 'Income'
        elif amount > 1000:  # Assume large positive amounts are likely income
            return 'Income'
    
    # Everything else with negative amount is expense
    if amount < 0:
        return 'Expense'
    
    return 'Other'

def calculate_budget_analysis(df, amount_col, desc_col, category_col):
    """Calculate comprehensive budget analysis."""
    # Clean the amount column - handle empty strings and convert to numeric
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
    
    # Remove rows with NaN amounts (like Opening Balance entries)
    df_clean = df.dropna(subset=[amount_col]).copy()
    
    # Create income/expense classification
    df_clean['Transaction_Type'] = df_clean.apply(lambda row: categorize_income_expense(
        row[desc_col], row[amount_col]), axis=1)
    
    # Calculate totals
    total_income = df_clean[df_clean['Transaction_Type'] == 'Income'][amount_col].sum()
    
    # For expenses, we want the absolute value since they're negative
    expense_data = df_clean[df_clean['Transaction_Type'] == 'Expense'].copy()
    expense_data['Abs_Amount'] = expense_data[amount_col].abs()
    
    total_expenses = expense_data['Abs_Amount'].sum()
    available_income = total_income - total_expenses
    
    # Category breakdown for expenses only
    category_breakdown = {}
    categories = ['Rent/Mortgage', 'Subscriptions', 'Dining Out', 'Transport', 'Groceries', 'Shopping', 'Other', 'Administrative']
    
    for category in categories:
        category_data = expense_data[expense_data[category_col] == category]
        category_total = category_data['Abs_Amount'].sum()
        category_percentage = (category_total / total_expenses * 100) if total_expenses > 0 else 0
        category_breakdown[category] = {
            'amount': category_total,
            'percentage': category_percentage,
            'count': len(category_data)
        }
    
    return {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'available_income': available_income,
        'category_breakdown': category_breakdown,
        'expense_data': expense_data
    }

def calculate_savings_annuity(monthly_savings, years_list=[1, 5, 10, 15, 20, 25], annual_return_rate=0.08):
    """
    Calculate annuity returns - how much you'll have by saving a fixed amount monthly.
    
    Args:
        monthly_savings: Fixed monthly amount to save (e.g. R100)
        years_list: List of years to calculate for
        annual_return_rate: Expected annual return rate (default 8%)
    
    Returns:
        Dictionary with simple annuity calculations for each time period
    """
    results = {}
    monthly_rate = annual_return_rate / 12
    
    for years in years_list:
        total_months = years * 12
        total_contributions = monthly_savings * total_months
        
        if monthly_rate == 0:
            # No growth scenario
            final_value = total_contributions
        else:
            # Future Value of Ordinary Annuity formula: PMT × [((1 + r)^n - 1) / r]
            final_value = monthly_savings * (((1 + monthly_rate) ** total_months - 1) / monthly_rate)
        
        total_interest_earned = final_value - total_contributions
        
        results[years] = {
            'monthly_payment': monthly_savings,
            'total_months': total_months,
            'total_contributions': total_contributions,
            'final_value': final_value,
            'interest_earned': total_interest_earned,
            'effective_return': (total_interest_earned / total_contributions * 100) if total_contributions > 0 else 0
        }
    
    return results

def calculate_compound_growth(monthly_payment, annual_rate, years):
    """Calculate compound growth with monthly contributions."""
    monthly_rate = annual_rate / 12
    total_months = years * 12
    
    if monthly_rate == 0:
        return monthly_payment * total_months
    
    # Future value of annuity formula
    future_value = monthly_payment * (((1 + monthly_rate) ** total_months - 1) / monthly_rate)
    return future_value

def generate_cost_cutting_suggestions(category_breakdown, total_expenses):
    """Generate realistic cost-cutting suggestions for each category."""
    suggestions = {}
    
    for category, data in category_breakdown.items():
        amount = data['amount']
        percentage = data['percentage']
        count = data['count']
        
        if amount == 0:
            suggestions[category] = "No expenses in this category."
            continue
        
        category_suggestions = []
        potential_savings = 0
        
        if category == 'Rent/Mortgage':
            if percentage > 35:
                category_suggestions.append("Consider finding a cheaper room share or moving to a less expensive area")
                potential_savings = amount * 0.15  # 15% savings
            elif percentage > 25:
                category_suggestions.append("Look for room sharing opportunities to split costs")
                potential_savings = amount * 0.10  # 10% savings
            else:
                category_suggestions.append("Rent is within recommended range (25-35% of income)")
        
        elif category == 'Subscriptions':
            if percentage > 15:
                category_suggestions.append("Cancel unused subscriptions and switch to cheaper mobile plans")
                potential_savings = amount * 0.25  # 25% savings
            else:
                category_suggestions.append("Review and cancel any unused services")
                potential_savings = amount * 0.15  # 15% savings
        
        elif category == 'Dining Out':
            if percentage > 15:
                category_suggestions.append("Limit takeaways to once per week, cook more meals at home")
                potential_savings = amount * 0.40  # 40% savings
            elif percentage > 10:
                category_suggestions.append("Reduce takeaways by half, meal prep on weekends")
                potential_savings = amount * 0.30  # 30% savings
            else:
                category_suggestions.append("Continue current dining habits or look for special offers")
                potential_savings = amount * 0.10  # 10% savings
        
        elif category == 'Transport':
            if percentage > 20:
                category_suggestions.append("Consider carpooling, walking for short distances, or monthly taxi passes")
                potential_savings = amount * 0.20  # 20% savings
            else:
                category_suggestions.append("Look for discounted transport options or walk when possible")
                potential_savings = amount * 0.10  # 10% savings
        
        elif category == 'Groceries':
            if percentage > 20:
                category_suggestions.append("Shop at cheaper stores, buy generic brands, meal plan weekly")
                potential_savings = amount * 0.20  # 20% savings
            else:
                category_suggestions.append("Use store loyalty programs and buy items on special")
                potential_savings = amount * 0.10  # 10% savings
        
        elif category == 'Shopping':
            if percentage > 10:
                category_suggestions.append("Implement a 'needs vs wants' rule, shop second-hand for clothing")
                potential_savings = amount * 0.35  # 35% savings
            else:
                category_suggestions.append("Continue current shopping habits, look for sales")
                potential_savings = amount * 0.15  # 15% savings
        
        else:  # Other/Administrative
            category_suggestions.append("Review all miscellaneous expenses for potential savings")
            potential_savings = amount * 0.10  # 10% savings
        
        suggestions[category] = {
            'suggestions': category_suggestions,
            'potential_savings': potential_savings,
            'current_amount': amount
        }
    
    return suggestions

def generate_budget_report(filepath):
    """Generate a comprehensive budget report for a single CSV file."""
    try:
        # Read the categorized CSV file
        df = pd.read_csv(filepath)
        
        # Find required columns
        amount_col = find_amount_column(df)
        desc_col = find_description_column(df)
        
        if not amount_col or not desc_col:
            print(f"❌ Required columns not found in {filepath}")
            return None
        
        # Ensure Category column exists
        if 'Category' not in df.columns:
            print(f"❌ Category column not found in {filepath}")
            return None
        
        category_col = 'Category'
        
        # Calculate budget analysis
        analysis = calculate_budget_analysis(df, amount_col, desc_col, category_col)
        
        # NEW: Try to use enhanced suggestions if available, otherwise use original
        enhanced_mode = False
        action_plan = None
        
        if ENHANCED_AVAILABLE:
            try:
                enhanced_suggestions, total_potential_savings, action_plan = generate_enhanced_cost_cutting_suggestions(
                    analysis['category_breakdown'], 
                    analysis['total_income'],
                    analysis['total_expenses']
                )
                suggestions = enhanced_suggestions
                enhanced_mode = True
                print(f"✅ Using enhanced suggestions for {os.path.basename(filepath)}")
            except Exception as e:
                print(f"⚠️ Enhanced suggestions failed for {os.path.basename(filepath)}, using basic: {e}")
                suggestions = generate_cost_cutting_suggestions(analysis['category_breakdown'], analysis['total_expenses'])
                total_potential_savings = sum(s.get('potential_savings', 0) for s in suggestions.values() if isinstance(s, dict))
        else:
            # Use original cost cutting suggestions
            suggestions = generate_cost_cutting_suggestions(analysis['category_breakdown'], analysis['total_expenses'])
            total_potential_savings = sum(s.get('potential_savings', 0) for s in suggestions.values() if isinstance(s, dict))
        
        # Create report content (keeping your original format)
        filename = os.path.basename(filepath).replace('categorized_', '').replace('.csv', '')
        report_content = f"""
# BUDGET ANALYSIS REPORT
## {filename.upper()}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

═══════════════════════════════════════════════════════════════

## 💰 FINANCIAL SUMMARY

**Monthly Income:**     R {analysis['total_income']:,.2f}
**Total Expenses:**     R {analysis['total_expenses']:,.2f}
**Available Income:**   R {analysis['available_income']:,.2f}

**Savings Rate:** {(analysis['available_income'] / analysis['total_income'] * 100) if analysis['total_income'] > 0 else 0:.1f}%

═══════════════════════════════════════════════════════════════

## 📊 EXPENSE BREAKDOWN BY CATEGORY

"""
        
        # Add category breakdown
        for category, data in analysis['category_breakdown'].items():
            if data['amount'] > 0:
                report_content += f"**{category}:** R {data['amount']:,.2f} ({data['percentage']:.1f}%)\n"
        
        report_content += "\n═══════════════════════════════════════════════════════════════\n\n"
        
        # NEW: Add enhanced sections if available
        if enhanced_mode and action_plan:
            report_content += "## 🎯 SMART BUDGET OPTIMIZATION\n"
            report_content += "### Based on South African household spending patterns\n\n"
            
            # Add action plan sections
            if action_plan.get('immediate_actions'):
                report_content += "**🔥 IMMEDIATE ACTIONS (High Impact):**\n"
                for action in action_plan['immediate_actions']:
                    report_content += f"• {action}\n"
                report_content += "\n"
            
            if action_plan.get('short_term_goals'):
                report_content += "**📅 SHORT-TERM GOALS (1-3 months):**\n"
                for action in action_plan['short_term_goals']:
                    report_content += f"• {action}\n"
                report_content += "\n"
            
            if action_plan.get('long_term_goals'):
                report_content += "**🎯 LONG-TERM GOALS (3+ months):**\n"
                for action in action_plan['long_term_goals']:
                    report_content += f"• {action}\n"
                report_content += "\n"
            
            report_content += "═══════════════════════════════════════════════════════════════\n\n"
            report_content += "## 💡 DETAILED OPTIMIZATION BREAKDOWN\n\n"
        else:
            report_content += "## 💡 BUDGET OPTIMIZATION SUGGESTIONS\n\n"
        
        # Add cost-cutting suggestions (works for both enhanced and original)
        for category, suggestion_data in suggestions.items():
            if suggestion_data != "No expenses in this category." and isinstance(suggestion_data, dict):
                if suggestion_data['current_amount'] > 0:
                    # NEW: Add priority emoji if enhanced mode
                    if enhanced_mode and 'priority' in suggestion_data:
                        priority_emoji = "🔥" if suggestion_data['priority'] >= 4 else "📅" if suggestion_data['priority'] >= 3 else "🎯"
                        report_content += f"### {priority_emoji} {category}\n"
                    else:
                        report_content += f"### {category}\n"
                    
                    report_content += f"Current: R {suggestion_data['current_amount']:,.2f}\n"
                    report_content += f"Potential Savings: R {suggestion_data['potential_savings']:,.2f}\n"
                    
                    # NEW: Add confidence level if enhanced mode
                    if enhanced_mode and 'confidence_level' in suggestion_data:
                        report_content += f"Confidence: {suggestion_data['confidence_level']}\n"
                    
                    # Handle both list and string suggestions
                    suggestion_list = suggestion_data['suggestions']
                    if isinstance(suggestion_list, str):
                        suggestion_list = [suggestion_list]
                    elif not isinstance(suggestion_list, list):
                        suggestion_list = [str(suggestion_list)]
                    
                    for suggestion in suggestion_list:
                        report_content += f"• {suggestion}\n"
                    report_content += "\n"
        
        report_content += "═══════════════════════════════════════════════════════════════\n\n"
        report_content += f"## 🎯 TOTAL POTENTIAL MONTHLY SAVINGS: R {total_potential_savings:,.2f}\n"
        report_content += f"**New Available Income:** R {analysis['available_income'] + total_potential_savings:,.2f}\n"
        report_content += f"**Improved Savings Rate:** {((analysis['available_income'] + total_potential_savings) / analysis['total_income'] * 100) if analysis['total_income'] > 0 else 0:.1f}%\n\n"
        
        # Add Savings Annuity Calculator (keeping your original section)
        report_content += "═══════════════════════════════════════════════════════════════\n\n"
        report_content += "## 💰 SAVINGS GROWTH CALCULATOR\n"
        report_content += "### Monthly Savings Annuity Projections\n\n"
        
        # Calculate annuity returns for current available income and potential savings
        current_monthly_savings = max(0, analysis['available_income'])
        optimized_monthly_savings = max(0, analysis['available_income'] + total_potential_savings)
        
        if current_monthly_savings > 0:
            report_content += f"**Scenario A: Current Available Income (R {current_monthly_savings:,.2f}/month)**\n\n"
            current_annuity_results = calculate_savings_annuity(current_monthly_savings)
            
            report_content += "| Years | Monthly Savings | Total Saved | Final Value | Interest Earned |\n"
            report_content += "|-------|----------------|-------------|-------------|----------------|\n"
            
            for years, result in current_annuity_results.items():
                report_content += f"| {years:2d} | R {result['monthly_payment']:,.2f} | R {result['total_contributions']:,.2f} | R {result['final_value']:,.2f} | R {result['interest_earned']:,.2f} |\n"
            
            report_content += "\n"
        
        if optimized_monthly_savings > current_monthly_savings:
            report_content += f"**Scenario B: With Cost Optimizations (R {optimized_monthly_savings:,.2f}/month)**\n\n"
            optimized_annuity_results = calculate_savings_annuity(optimized_monthly_savings)
            
            report_content += "| Years | Monthly Savings | Total Saved | Final Value | Interest Earned | Extra vs Current |\n"
            report_content += "|-------|----------------|-------------|-------------|----------------|------------------|\n"
            
            for years, result in optimized_annuity_results.items():
                current_result = current_annuity_results.get(years, {'final_value': 0}) if current_monthly_savings > 0 else {'final_value': 0}
                extra_value = result['final_value'] - current_result['final_value']
                
                report_content += f"| {years:2d} | R {result['monthly_payment']:,.2f} | R {result['total_contributions']:,.2f} | R {result['final_value']:,.2f} | R {result['interest_earned']:,.2f} | R {extra_value:,.2f} |\n"
            
            report_content += "\n"
        
        # Add some key insights (keeping your original)
        if current_monthly_savings > 0:
            # Get 10-year result for insights
            ten_year_result = current_annuity_results.get(10, {})
            twenty_year_result = current_annuity_results.get(20, {})
            
            if ten_year_result:
                report_content += "**💡 Key Insights:**\n"
                report_content += f"• After 10 years: You'll have R {ten_year_result['final_value']:,.2f} (R {ten_year_result['interest_earned']:,.2f} in interest)\n"
                
                if twenty_year_result:
                    report_content += f"• After 20 years: You'll have R {twenty_year_result['final_value']:,.2f} (R {twenty_year_result['interest_earned']:,.2f} in interest)\n"
                    double_time = twenty_year_result['final_value'] / ten_year_result['final_value'] if ten_year_result['final_value'] > 0 else 0
                    report_content += f"• Your money grows {double_time:.1f}x from year 10 to year 20 due to compound interest!\n"
                
                report_content += "\n"
        
        # Add calculation assumptions (keeping your original)
        report_content += "**Calculation Assumptions:**\n"
        report_content += "• 8% annual return (compounded monthly)\n"
        report_content += "• Fixed monthly contributions at month-end\n"
        report_content += "• No taxes considered (use TFSA or retirement annuity for tax benefits)\n"
        report_content += "• Returns are estimates based on historical averages\n\n"
        
        # Financial health assessment (keeping your original)
        savings_rate = (analysis['available_income'] / analysis['total_income'] * 100) if analysis['total_income'] > 0 else 0
        
        report_content += "## 📈 FINANCIAL HEALTH ASSESSMENT\n\n"
        
        if savings_rate >= 20:
            report_content += "🟢 **Excellent:** You're saving over 20% of your income!\n"
        elif savings_rate >= 10:
            report_content += "🟡 **Good:** You're saving 10-20% of your income. Room for improvement.\n"
        elif savings_rate >= 0:
            report_content += "🟠 **Caution:** Low savings rate. Focus on expense reduction.\n"
        else:
            report_content += "🔴 **Alert:** Spending more than you earn. Immediate action required.\n"
        
        # NEW: Return additional data for API endpoints
        return {
            'content': report_content,
            'filename': filename,
            'analysis': analysis,
            'suggestions': suggestions,
            'total_potential_savings': total_potential_savings,
            'enhanced_mode': enhanced_mode,
            'action_plan': action_plan
        }
        
    except Exception as e:
        print(f"❌ Error generating report for {filepath}: {e}")
        return None

def main():
    """Process all categorized CSV files and generate budget reports."""
    print("📊 Budget Analysis Report Generator")
    if ENHANCED_AVAILABLE:
        print("🚀 Enhanced optimization features available!")
    print("=" * 60)
    
    # Find categorized CSV files
    csv_files = glob.glob(os.path.join(CATEGORIZED_DATA_DIR, "categorized_*.csv"))
    
    if not csv_files:
        print(f"❌ No categorized CSV files found in {CATEGORIZED_DATA_DIR}")
        print("Please run the bank categorizer first!")
        return
    
    print(f"📁 Found {len(csv_files)} categorized file(s) to analyze:")
    for file in csv_files:
        print(f"  📄 {os.path.basename(file)}")
    print()
    
    # Generate reports for each file
    successful_reports = 0
    for filepath in csv_files:
        print(f"🔄 Processing {os.path.basename(filepath)}...")
        
        report = generate_budget_report(filepath)
        if report:
            # Save report as text file
            report_filename = f"budget_report_{report['filename']}.txt"
            report_path = os.path.join(REPORTS_DIR, report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report['content'])
            
            enhancement_note = " (Enhanced)" if report.get('enhanced_mode') else ""
            print(f"✅ Report saved: {report_filename}{enhancement_note}")
            successful_reports += 1
        else:
            print(f"❌ Failed to generate report for {os.path.basename(filepath)}")
        print()
    
    print("=" * 60)
    print(f"🎉 Report generation complete!")
    print(f"✅ Successfully generated: {successful_reports}/{len(csv_files)} reports")
    print(f"📂 Reports saved to: {REPORTS_DIR}")
    
    # Show sample report content for first file (if any)
    if successful_reports > 0:
        print("\n" + "=" * 60)
        print("📋 SAMPLE REPORT PREVIEW:")
        print("=" * 60)
        sample_file = csv_files[0]
        sample_report = generate_budget_report(sample_file)
        if sample_report:
            # Show first 25 lines of the report
            lines = sample_report['content'].split('\n')[:25]
            print('\n'.join(lines))
            if len(sample_report['content'].split('\n')) > 25:
                print("...")
                print("[Full report saved to file]")

if __name__ == "__main__":
    main()