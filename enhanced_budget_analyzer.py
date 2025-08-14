"""
Enhanced budget analyzer with protected categories system and debt/investment split recommendations
UPDATED VERSION - Integrates your friend's improvements with debt/investment split support
"""
import pandas as pd
import glob
import os
from datetime import datetime
import re
from typing import Dict, List, Tuple, Set

# Import existing modules
try:
    from config import DATA_DIRECTORY, OUTPUT_DIRECTORY
except ImportError:
    DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")
    OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "output")

# Configuration for protected categories
PROTECTED_CATEGORIES = {
    'Debt Repayment', 'Loan Repayment', 'Bond/Mortgage', 'Credit Card Payment',
    'Vehicle Finance', 'Student Loan', 'Debt', 'Bond Repayment'
}

PROTECTED_KEYWORDS = [
    'loan', 'repayment', 'installment', 'instalment', 'bond', 'mortgage',
    'credit card', 'car finance', 'vehicle finance', 'hire purchase',
    'arrears', 'debt', 'student loan', 'monthly payment'
]

class EnhancedSouthAfricanBudgetOptimizer:
    """Enhanced budget optimizer with protected categories support."""
    
    def __init__(self):
        # South African specific income brackets (monthly, ZAR)
        self.income_brackets = {
            'very_low': (0, 3500),
            'low': (3500, 8000),
            'low_middle': (8000, 15000),
            'middle': (15000, 25000),
            'upper_middle': (25000, float('inf'))
        }
        
        # Realistic expense ratios for SA households by income bracket
        self.recommended_ratios = {
            'very_low': {
                'housing': 0.40, 'food': 0.35, 'transport': 0.15,
                'utilities': 0.08, 'other': 0.02
            },
            'low': {
                'housing': 0.35, 'food': 0.30, 'transport': 0.15,
                'utilities': 0.10, 'other': 0.10
            },
            'low_middle': {
                'housing': 0.30, 'food': 0.25, 'transport': 0.12,
                'utilities': 0.08, 'other': 0.25
            },
            'middle': {
                'housing': 0.25, 'food': 0.20, 'transport': 0.10,
                'utilities': 0.07, 'other': 0.38
            }
        }
    
    def determine_income_bracket(self, monthly_income: float) -> str:
        """Determine income bracket based on monthly income."""
        for bracket, (min_val, max_val) in self.income_brackets.items():
            if min_val <= monthly_income < max_val:
                return bracket
        return 'upper_middle'
    
    def calculate_weighted_savings_potential(self, category_breakdown: Dict, 
                                           reducible_category_breakdown: Dict,
                                           protected_categories_present: Set,
                                           total_income: float, 
                                           total_expenses: float) -> Tuple[Dict, float]:
        """
        Calculate savings potential using weighted analysis with protected categories.
        Only reducible amounts are considered for optimization.
        """
        income_bracket = self.determine_income_bracket(total_income)
        recommended = self.recommended_ratios.get(income_bracket, self.recommended_ratios['low'])
        
        savings_analysis = {}
        total_potential_savings = 0
        
        # Map categories to our standard categories
        category_mapping = {
            'Rent/Mortgage': 'housing',
            'Groceries': 'food', 
            'Transport': 'transport',
            'Subscriptions': 'utilities',
            'Dining Out': 'food',
            'Shopping': 'other',
            'Other': 'other',
            'Administrative': 'other'
        }
        
        # Combine related categories (using reducible amounts only)
        consolidated_spending = {
            'housing': 0, 'food': 0, 'transport': 0, 'utilities': 0, 'other': 0
        }
        
        for category, data in reducible_category_breakdown.items():
            mapped_cat = category_mapping.get(category, 'other')
            consolidated_spending[mapped_cat] += data.get('amount', 0)
        
        # Calculate savings for each consolidated category
        for category, current_amount in consolidated_spending.items():
            if current_amount == 0:
                continue
                
            current_ratio = current_amount / total_expenses if total_expenses > 0 else 0
            recommended_ratio = recommended.get(category, 0.10)
            recommended_amount = total_expenses * recommended_ratio
            
            savings_analysis[category] = self._calculate_category_savings(
                category, current_amount, recommended_amount, current_ratio, 
                recommended_ratio, income_bracket, total_income
            )
            
            total_potential_savings += savings_analysis[category]['potential_savings']
        
        return savings_analysis, total_potential_savings
    
    def _calculate_category_savings(self, category: str, current_amount: float, 
                                  recommended_amount: float, current_ratio: float, 
                                  recommended_ratio: float, income_bracket: str, 
                                  total_income: float) -> Dict:
        """Calculate category-specific savings with weighted factors."""
        
        # Base savings potential
        if current_amount > recommended_amount:
            base_savings = current_amount - recommended_amount
        else:
            base_savings = current_amount * 0.05  # 5% optimization
        
        # Apply category-specific multipliers and constraints
        if category == 'housing':
            difficulty_multiplier = {
                'very_low': 0.05, 'low': 0.10, 'low_middle': 0.15, 'middle': 0.20
            }.get(income_bracket, 0.10)
            
            potential_savings = base_savings * difficulty_multiplier
            strategies = self._get_housing_strategies(income_bracket, current_ratio)
            
        elif category == 'food':
            if current_ratio > recommended_ratio * 1.3:
                potential_savings = base_savings * 0.25
                strategies = [
                    "Shop at cheaper stores (Shoprite, Pick n Pay basics range)",
                    "Buy generic/home brands instead of name brands",
                    "Plan weekly meals and create shopping lists",
                    "Buy bulk items for staples (rice, maize meal, lentils)",
                    "Reduce meat consumption, increase affordable proteins like eggs/beans"
                ]
            elif current_ratio > recommended_ratio * 1.1:
                potential_savings = base_savings * 0.15
                strategies = [
                    "Switch to more affordable stores occasionally",
                    "Use store loyalty programs and specials",
                    "Meal plan to reduce food waste",
                    "Buy seasonal vegetables and fruits"
                ]
            else:
                potential_savings = current_amount * 0.05
                strategies = [
                    "Continue current habits, look for monthly specials",
                    "Use store apps for digital coupons"
                ]
        
        elif category == 'transport':
            if income_bracket in ['very_low', 'low']:
                potential_savings = base_savings * 0.10
                strategies = [
                    "Walk for short distances when safe",
                    "Use monthly taxi passes if available",
                    "Consider carpooling for regular routes"
                ]
            else:
                potential_savings = base_savings * 0.20
                strategies = [
                    "Consider public transport alternatives",
                    "Carpool with colleagues",
                    "Combine trips to reduce frequency",
                    "Negotiate group transport rates"
                ]
        
        elif category == 'utilities':
            potential_savings = base_savings * 0.30
            strategies = [
                "Review and cancel unused subscriptions",
                "Switch to cheaper mobile plans (prepaid vs contract)",
                "Use prepaid electricity to monitor usage",
                "Negotiate better rates with service providers",
                "Share streaming subscriptions with family/friends"
            ]
        
        else:  # other
            potential_savings = base_savings * 0.35
            strategies = [
                "Implement 'needs vs wants' evaluation before purchases",
                "Set monthly discretionary spending limit",
                "Find free/low-cost entertainment alternatives",
                "Shop second-hand for clothing and electronics"
            ]
        
        # Apply income-based reality check
        income_constraint = {
            'very_low': 0.7, 'low': 0.8, 'low_middle': 0.9, 'middle': 1.0
        }.get(income_bracket, 1.0)
        
        potential_savings *= income_constraint
        
        # Ensure savings don't exceed reasonable limits
        max_category_reduction = current_amount * 0.4
        potential_savings = min(potential_savings, max_category_reduction)
        
        return {
            'current_amount': current_amount,
            'recommended_amount': recommended_amount,
            'potential_savings': potential_savings,
            'savings_percentage': (potential_savings / current_amount * 100) if current_amount > 0 else 0,
            'strategies': strategies,
            'priority': self._calculate_priority(current_ratio, recommended_ratio, potential_savings)
        }
    
    def _get_housing_strategies(self, income_bracket: str, current_ratio: float) -> List[str]:
        """Get housing-specific strategies based on income and current spending."""
        if current_ratio > 0.40:
            if income_bracket in ['very_low', 'low']:
                return [
                    "Look for room sharing opportunities",
                    "Consider moving to cheaper areas (if transport costs don't offset)",
                    "Investigate RDP housing options if eligible"
                ]
            else:
                return [
                    "Find roommates to share costs",
                    "Consider relocating to more affordable areas",
                    "Negotiate rent reduction for longer lease terms"
                ]
        elif current_ratio > 0.30:
            return [
                "Look for opportunities to share accommodation costs",
                "Negotiate with landlord for minor rent reduction"
            ]
        else:
            return ["Housing costs are reasonable for your income level"]
    
    def _calculate_priority(self, current_ratio: float, recommended_ratio: float, 
                          potential_savings: float) -> int:
        """Calculate optimization priority (1-5, 5 being highest priority)."""
        ratio_excess = max(0, current_ratio - recommended_ratio)
        
        if potential_savings > 200 and ratio_excess > 0.10:
            return 5
        elif potential_savings > 100 and ratio_excess > 0.05:
            return 4
        elif potential_savings > 50:
            return 3
        elif potential_savings > 20:
            return 2
        else:
            return 1

def find_amount_column(df: pd.DataFrame) -> str:
    """Find the amount column with different possible names."""
    possible_names = [
        'Amount', 'amount', 'AMOUNT', 'Amount (ZAR)', 'Amount(ZAR)', 'amount_zar',
        'Debit', 'debit', 'DEBIT', 'Credit', 'credit', 'CREDIT',
        'Transaction Amount', 'transaction_amount', 'Value', 'value', 'VALUE'
    ]
    
    for col in df.columns:
        if col in possible_names:
            return col
        col_lower = col.lower()
        if any(name.lower() in col_lower for name in ['amount', 'debit', 'credit', 'value']):
            return col
    
    return None

def find_description_column(df: pd.DataFrame) -> str:
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

def categorize_income_expense(description: str, amount: float) -> str:
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
    
    # Check if it's likely income
    if amount > 0:
        if any(keyword in desc_lower for keyword in income_keywords):
            return 'Income'
        elif amount > 1000:  # Assume large positive amounts are likely income
            return 'Income'
    
    # Everything else with negative amount is expense
    if amount < 0:
        return 'Expense'
    
    return 'Other'

def calculate_enhanced_budget_analysis(df: pd.DataFrame, amount_col: str, 
                                     desc_col: str, category_col: str) -> Dict:
    """Calculate comprehensive budget analysis with reducible vs protected split."""
    # Clean the amount column
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
    df_clean = df.dropna(subset=[amount_col]).copy()

    # Create income/expense classification
    df_clean['Transaction_Type'] = df_clean.apply(
        lambda row: categorize_income_expense(row[desc_col], row[amount_col]), axis=1
    )

    # Calculate totals
    total_income = df_clean[df_clean['Transaction_Type'] == 'Income'][amount_col].sum()

    # For expenses, we want the absolute value since they're negative
    expense_data = df_clean[df_clean['Transaction_Type'] == 'Expense'].copy()
    expense_data['Abs_Amount'] = expense_data[amount_col].abs()

    # Mark which expense rows are reducible
    expense_data['ReduceAllowed'] = True

    # CSV override: if a column 'ReduceAllowed' exists, respect it
    if 'ReduceAllowed' in df.columns:
        def _coerce_bool(v):
            if isinstance(v, str):
                return v.strip().lower() in ('1', 'true', 'yes', 'y')
            return bool(v)
        expense_data['ReduceAllowed'] = df.loc[expense_data.index, 'ReduceAllowed'].map(_coerce_bool)

    # Category-based lock
    if category_col in expense_data.columns:
        expense_data.loc[expense_data[category_col].isin(PROTECTED_CATEGORIES), 'ReduceAllowed'] = False

    # Keyword-based lock on description
    desc_lower = expense_data[desc_col].astype(str).str.lower()
    kw_mask = False
    for kw in PROTECTED_KEYWORDS:
        if isinstance(kw_mask, bool):
            kw_mask = desc_lower.str.contains(kw, na=False)
        else:
            kw_mask = kw_mask | desc_lower.str.contains(kw, na=False)
    expense_data.loc[kw_mask, 'ReduceAllowed'] = False

    total_expenses = expense_data['Abs_Amount'].sum()
    available_income = total_income - total_expenses

    # Standard category breakdown (everything, for display)
    categories = ['Rent/Mortgage', 'Subscriptions', 'Dining Out', 'Transport', 
                  'Groceries', 'Shopping', 'Other', 'Administrative']
    category_breakdown = {}
    for category in categories:
        category_data = expense_data[expense_data[category_col] == category] if category_col in expense_data.columns else expense_data.iloc[0:0]
        category_total = category_data['Abs_Amount'].sum()
        category_percentage = (category_total / total_expenses * 100) if total_expenses > 0 else 0
        category_breakdown[category] = {
            'amount': category_total,
            'percentage': category_percentage,
            'count': len(category_data)
        }

    # Reducible-only category breakdown (used for optimization)
    reducible_category_breakdown = {}
    for category in categories:
        if category_col in expense_data.columns:
            cat_rows = expense_data[(expense_data[category_col] == category) & (expense_data['ReduceAllowed'] == True)]
        else:
            cat_rows = expense_data.iloc[0:0]
        reducible_total = cat_rows['Abs_Amount'].sum()
        reducible_category_breakdown[category] = {
            'amount': reducible_total,
            'count': len(cat_rows)
        }

    # Which categories contain any protected (locked) rows
    if category_col in expense_data.columns:
        protected_categories_present = set(
            expense_data.loc[(expense_data['ReduceAllowed'] == False), category_col].dropna().unique().tolist()
        )
    else:
        protected_categories_present = set()

    # Add transactions to the return data
    transactions = df_clean.to_dict('records')

    return {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'available_income': available_income,
        'category_breakdown': category_breakdown,
        'reducible_category_breakdown': reducible_category_breakdown,
        'protected_categories_present': protected_categories_present,
        'expense_data': expense_data,
        'transactions': transactions  # NEW: Include transactions for debt payment detection
    }

def generate_enhanced_cost_cutting_suggestions(category_breakdown: Dict,
                                             reducible_category_breakdown: Dict,
                                             protected_categories_present: Set,
                                             total_income: float,
                                             total_expenses: float) -> Tuple[Dict, float, Dict]:
    """
    Enhanced version using weighted analysis for South African context.
    Savings are computed ONLY from reducible portions (locked items excluded).
    """
    optimizer = EnhancedSouthAfricanBudgetOptimizer()

    # Build a synthetic breakdown using reducible-only amounts
    reducible_breakdown_for_optimizer = {}
    for cat, data in category_breakdown.items():
        reducible_amt = reducible_category_breakdown.get(cat, {}).get('amount', 0.0)
        reducible_breakdown_for_optimizer[cat] = {
            'amount': reducible_amt,
            'percentage': 0,
            'count': 0
        }

    savings_analysis, total_potential_savings = optimizer.calculate_weighted_savings_potential(
        reducible_breakdown_for_optimizer, 
        reducible_category_breakdown,
        protected_categories_present,
        total_income, 
        total_expenses
    )

    enhanced_suggestions = {}
    reverse_mapping = {
        'housing': ['Rent/Mortgage'],
        'food': ['Groceries', 'Dining Out'],
        'transport': ['Transport'],
        'utilities': ['Subscriptions'],
        'other': ['Shopping', 'Other', 'Administrative']
    }

    for consolidated_cat, analysis in savings_analysis.items():
        original_categories = reverse_mapping.get(consolidated_cat, [consolidated_cat])

        for orig_cat in original_categories:
            current_amount_full = category_breakdown.get(orig_cat, {}).get('amount', 0.0)
            reducible_amount = reducible_category_breakdown.get(orig_cat, {}).get('amount', 0.0)

            if reducible_amount <= 0:
                strategies = analysis['strategies'] if isinstance(analysis['strategies'], list) else [analysis['strategies']]
                note = []
                if (orig_cat in protected_categories_present) or (orig_cat in PROTECTED_CATEGORIES):
                    note = ["🔒 Locked: fixed obligation (excluded from optimization)"]
                enhanced_suggestions[orig_cat] = {
                    'suggestions': (note or ["No discretionary reduction identified"]) + strategies[:1],
                    'potential_savings': 0.0,
                    'current_amount': current_amount_full,
                    'priority': 1,
                    'confidence_level': 'Low'
                }
                continue

            denom = analysis['current_amount'] if analysis['current_amount'] > 0 else 0.0
            proportion = (reducible_amount / denom) if denom > 0 else 0.0
            allocated_savings = analysis['potential_savings'] * proportion

            strategies = analysis['strategies'] if isinstance(analysis['strategies'], list) else [analysis['strategies']]

            enhanced_suggestions[orig_cat] = {
                'suggestions': strategies,
                'potential_savings': allocated_savings,
                'current_amount': current_amount_full,
                'priority': analysis['priority'],
                'confidence_level': 'High' if analysis['priority'] >= 4 else 'Medium' if analysis['priority'] >= 3 else 'Low'
            }

    sorted_suggestions = sorted(
        enhanced_suggestions.items(),
        key=lambda x: (x[1]['priority'], x[1]['potential_savings']),
        reverse=True
    )

    action_plan = {
        'immediate_actions': [],
        'short_term_goals': [],
        'long_term_goals': []
    }

    for category, suggestion in sorted_suggestions:
        strategies = suggestion['suggestions'] if isinstance(suggestion['suggestions'], list) else [suggestion['suggestions']]

        if suggestion['potential_savings'] <= 0:
            continue

        if suggestion['priority'] >= 4:
            action_plan['immediate_actions'].extend([f"{category}: {strategy}" for strategy in strategies[:2]])
        elif suggestion['priority'] >= 3:
            action_plan['short_term_goals'].extend([f"{category}: {strategy}" for strategy in strategies[:2]])
        else:
            action_plan['long_term_goals'].extend([f"{category}: {strategy}" for strategy in strategies[:1]])

    return enhanced_suggestions, total_potential_savings, action_plan

def calculate_savings_annuity(monthly_savings: float, years_list: List[int] = None, 
                            annual_return_rate: float = 0.08) -> Dict:
    """Calculate annuity returns for savings projections."""
    if years_list is None:
        years_list = [1, 5, 10, 15, 20, 25]
        
    results = {}
    monthly_rate = annual_return_rate / 12
    
    for years in years_list:
        total_months = years * 12
        total_contributions = monthly_savings * total_months
        
        if monthly_rate == 0:
            final_value = total_contributions
        else:
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

def extract_debt_payments_from_transactions(transactions: List) -> List:
    """Extract debt payment info from transaction list for debt-to-income ratio calculation."""
    debt_payments = []
    for transaction in transactions:
        if transaction.get('IsDebtPayment') and transaction.get('DebtName'):
            debt_payments.append({
                'name': transaction.get('DebtName'),
                'amount': abs(transaction.get('Amount (ZAR)', 0)),
                'type': transaction.get('DebtKind', 'unknown')
            })
    return debt_payments

def calculate_recommended_split(debt_to_income_ratio: float) -> Dict:
    """
    Calculate recommended debt/investment split based on debt burden.
    NEW FUNCTION for debt/investment allocation recommendations.
    """
    if debt_to_income_ratio > 0.4:  # High debt burden (>40% of income)
        return {
            "debt": 0.8, 
            "investment": 0.2, 
            "rationale": "High debt burden detected - prioritize aggressive debt elimination"
        }
    elif debt_to_income_ratio > 0.28:  # Moderate debt burden (28-40% of income)
        return {
            "debt": 0.6, 
            "investment": 0.4, 
            "rationale": "Moderate debt levels - balanced approach recommended"
        }
    elif debt_to_income_ratio > 0.15:  # Low debt burden (15-28% of income)
        return {
            "debt": 0.4, 
            "investment": 0.6, 
            "rationale": "Low debt burden - favor long-term investment growth"
        }
    else:  # Very low/no debt burden (<15% of income)
        return {
            "debt": 0.2, 
            "investment": 0.8, 
            "rationale": "Minimal debt burden - focus on wealth building through investment"
        }

def generate_enhanced_budget_report(filepath: str) -> Dict:
    """
    Generate enhanced budget report compatible with your existing Flask API.
    UPDATED VERSION - Integrates protected categories, weighted optimization, and debt/investment split recommendations.
    """
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
        
        # Calculate enhanced budget analysis
        analysis = calculate_enhanced_budget_analysis(df, amount_col, desc_col, category_col)
        
        # Generate enhanced cost-cutting suggestions
        enhanced_suggestions, total_potential_savings, action_plan = generate_enhanced_cost_cutting_suggestions(
            analysis['category_breakdown'],
            analysis.get('reducible_category_breakdown', {}),
            analysis.get('protected_categories_present', set()),
            analysis['total_income'],
            analysis['total_expenses']
        )
        
        # NEW: Calculate debt-to-income ratio for split recommendations
        debt_payments = extract_debt_payments_from_transactions(analysis.get('transactions', []))
        total_debt_payments = sum(payment['amount'] for payment in debt_payments)
        debt_to_income_ratio = total_debt_payments / analysis['total_income'] if analysis['total_income'] > 0 else 0
        
        # NEW: Add split recommendation logic
        recommended_split = calculate_recommended_split(debt_to_income_ratio)
        optimized_available_income = analysis['available_income'] + total_potential_savings
        
        return {
            'analysis': analysis,
            'suggestions': enhanced_suggestions,
            'total_potential_savings': total_potential_savings,
            'action_plan': action_plan,
            'enhanced_mode': True,
            'annuity_projection': calculate_savings_annuity(analysis['available_income']),
            'optimized_available_income': optimized_available_income,
            
            # NEW FIELDS for debt/investment split:
            'debt_to_income_ratio': debt_to_income_ratio,
            'total_debt_payments': total_debt_payments,
            'debt_payments_detected': debt_payments,
            'recommended_debt_ratio': recommended_split['debt'],
            'recommended_investment_ratio': recommended_split['investment'],
            'split_rationale': recommended_split['rationale'],
            'recommended_debt_budget': optimized_available_income * recommended_split['debt'],
            'recommended_investment_budget': optimized_available_income * recommended_split['investment'],
        }
        
    except Exception as e:
        print(f"❌ Error generating enhanced report for {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None