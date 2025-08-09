"""
enhanced_budget_analyzer.py - Enhanced budget analysis with South African context
Extends the existing budget_analyzer with weighted optimization techniques
"""

import pandas as pd
from typing import Dict, List, Tuple

class SouthAfricanBudgetOptimizer:
    """
    Enhanced budget optimizer using weighted techniques based on South African 
    low to low-middle income household realities.
    """
    
    def __init__(self):
        # South African specific income brackets (monthly, ZAR)
        self.income_brackets = {
            'very_low': (0, 3500),      # Below poverty line
            'low': (3500, 8000),        # Lower income
            'low_middle': (8000, 15000), # Lower middle class
            'middle': (15000, 25000),    # Middle class
            'upper_middle': (25000, float('inf'))  # Upper middle+
        }
        
        # Realistic expense ratios for SA households by income bracket
        self.recommended_ratios = {
            'very_low': {
                'housing': 0.40,  # Often higher due to limited options
                'food': 0.35,     # Higher food burden
                'transport': 0.15,
                'utilities': 0.08,
                'other': 0.02
            },
            'low': {
                'housing': 0.35,
                'food': 0.30,
                'transport': 0.15,
                'utilities': 0.10,
                'other': 0.10
            },
            'low_middle': {
                'housing': 0.30,
                'food': 0.25,
                'transport': 0.12,
                'utilities': 0.08,
                'other': 0.25
            },
            'middle': {
                'housing': 0.25,
                'food': 0.20,
                'transport': 0.10,
                'utilities': 0.07,
                'other': 0.38
            }
        }
    
    def determine_income_bracket(self, monthly_income: float) -> str:
        """Determine income bracket based on monthly income."""
        for bracket, (min_val, max_val) in self.income_brackets.items():
            if min_val <= monthly_income < max_val:
                return bracket
        return 'upper_middle'
    
    def calculate_weighted_savings_potential(self, category_breakdown: Dict, total_income: float, total_expenses: float) -> Tuple[Dict, float]:
        """
        Calculate savings potential using weighted analysis based on:
        1. Income bracket
        2. Current spending vs recommended ratios
        3. Category-specific optimization strategies
        4. South African context (transport costs, food prices, etc.)
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
            'Dining Out': 'food',  # Combine with groceries
            'Shopping': 'other',
            'Other': 'other',
            'Administrative': 'other'
        }
        
        # Combine related categories
        consolidated_spending = {
            'housing': 0,
            'food': 0,
            'transport': 0,
            'utilities': 0,
            'other': 0
        }
        
        for category, data in category_breakdown.items():
            mapped_cat = category_mapping.get(category, 'other')
            consolidated_spending[mapped_cat] += data['amount']
        
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
    
    def _calculate_category_savings(self, category: str, current_amount: float, recommended_amount: float, 
                                  current_ratio: float, recommended_ratio: float, income_bracket: str, total_income: float) -> Dict:
        """Calculate category-specific savings with weighted factors."""
        
        # Base savings potential
        if current_amount > recommended_amount:
            base_savings = current_amount - recommended_amount
        else:
            # Even if within recommended range, some optimization possible
            base_savings = current_amount * 0.05  # 5% optimization
        
        # Apply category-specific multipliers and constraints
        if category == 'housing':
            # Housing is hardest to reduce, especially for lower incomes
            difficulty_multiplier = {
                'very_low': 0.05,   # Very difficult to find cheaper
                'low': 0.10,        # Limited options
                'low_middle': 0.15, # Some flexibility
                'middle': 0.20      # More options available
            }.get(income_bracket, 0.10)
            
            potential_savings = base_savings * difficulty_multiplier
            strategies = self._get_housing_strategies(income_bracket, current_ratio)
            
        elif category == 'food':
            # Food savings vary by shopping habits and location
            if current_ratio > recommended_ratio * 1.3:
                potential_savings = base_savings * 0.25  # 25% if way overspending
                strategies = [
                    "Shop at cheaper stores (Shoprite, Pick n Pay basics range)",
                    "Buy generic/home brands instead of name brands",
                    "Plan weekly meals and create shopping lists",
                    "Buy bulk items for staples (rice, maize meal, lentils)",
                    "Reduce meat consumption, increase affordable proteins like eggs/beans"
                ]
            elif current_ratio > recommended_ratio * 1.1:
                potential_savings = base_savings * 0.15  # 15% if moderately overspending
                strategies = [
                    "Switch to more affordable stores occasionally",
                    "Use store loyalty programs and specials",
                    "Meal plan to reduce food waste",
                    "Buy seasonal vegetables and fruits"
                ]
            else:
                potential_savings = current_amount * 0.05  # 5% through better deals
                strategies = [
                    "Continue current habits, look for monthly specials",
                    "Use store apps for digital coupons"
                ]
        
        elif category == 'transport':
            # Transport costs are high in SA, but some optimization possible
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
            # Includes subscriptions, airtime, electricity
            potential_savings = base_savings * 0.30  # Good potential here
            strategies = [
                "Review and cancel unused subscriptions",
                "Switch to cheaper mobile plans (prepaid vs contract)",
                "Use prepaid electricity to monitor usage",
                "Negotiate better rates with service providers",
                "Share streaming subscriptions with family/friends"
            ]
        
        else:  # other
            # Discretionary spending - highest savings potential
            potential_savings = base_savings * 0.35
            strategies = [
                "Implement 'needs vs wants' evaluation before purchases",
                "Set monthly discretionary spending limit",
                "Find free/low-cost entertainment alternatives",
                "Shop second-hand for clothing and electronics"
            ]
        
        # Apply income-based reality check
        # Lower incomes have less room for optimization
        income_constraint = {
            'very_low': 0.7,    # Very limited optimization room
            'low': 0.8,         # Some constraints
            'low_middle': 0.9,  # Moderate flexibility
            'middle': 1.0       # Full potential
        }.get(income_bracket, 1.0)
        
        potential_savings *= income_constraint
        
        # Ensure savings don't exceed reasonable limits
        max_category_reduction = current_amount * 0.4  # Never more than 40% reduction
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
        if current_ratio > 0.40:  # Spending more than 40% on housing
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
    
    def _calculate_priority(self, current_ratio: float, recommended_ratio: float, potential_savings: float) -> int:
        """Calculate optimization priority (1-5, 5 being highest priority)."""
        ratio_excess = max(0, current_ratio - recommended_ratio)
        
        if potential_savings > 200 and ratio_excess > 0.10:
            return 5  # High priority - significant overspending
        elif potential_savings > 100 and ratio_excess > 0.05:
            return 4  # Medium-high priority
        elif potential_savings > 50:
            return 3  # Medium priority
        elif potential_savings > 20:
            return 2  # Low-medium priority
        else:
            return 1  # Low priority

def generate_enhanced_cost_cutting_suggestions(category_breakdown: Dict, total_income: float, total_expenses: float) -> Tuple[Dict, float, Dict]:
    """
    Enhanced version using weighted analysis for South African context.
    
    Args:
        category_breakdown: Original category breakdown from existing code
        total_income: Monthly income
        total_expenses: Total monthly expenses
    
    Returns:
        Tuple of (enhanced_suggestions, total_potential_savings, action_plan)
    """
    optimizer = SouthAfricanBudgetOptimizer()
    
    # Get weighted analysis
    savings_analysis, total_potential_savings = optimizer.calculate_weighted_savings_potential(
        category_breakdown, total_income, total_expenses
    )
    
    # Convert back to format similar to original function for compatibility
    enhanced_suggestions = {}
    
    # Map back to original categories
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
            if orig_cat in category_breakdown and category_breakdown[orig_cat]['amount'] > 0:
                # Proportionally allocate savings if multiple original categories map to one
                proportion = category_breakdown[orig_cat]['amount'] / analysis['current_amount'] if analysis['current_amount'] > 0 else 0
                allocated_savings = analysis['potential_savings'] * proportion
                
                # Ensure strategies is always a list
                strategies = analysis['strategies'] if isinstance(analysis['strategies'], list) else [analysis['strategies']]
                
                enhanced_suggestions[orig_cat] = {
                    'suggestions': strategies,
                    'potential_savings': allocated_savings,
                    'current_amount': category_breakdown[orig_cat]['amount'],
                    'priority': analysis['priority'],
                    'confidence_level': 'High' if analysis['priority'] >= 4 else 'Medium' if analysis['priority'] >= 3 else 'Low'
                }
    
    # Create action plan
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
        # Ensure suggestions is always a list for processing
        strategies = suggestion['suggestions'] if isinstance(suggestion['suggestions'], list) else [suggestion['suggestions']]
        
        if suggestion['priority'] >= 4:
            action_plan['immediate_actions'].extend([
                f"{category}: {strategy}" for strategy in strategies[:2]
            ])
        elif suggestion['priority'] >= 3:
            action_plan['short_term_goals'].extend([
                f"{category}: {strategy}" for strategy in strategies[:2]
            ])
        else:
            action_plan['long_term_goals'].extend([
                f"{category}: {strategy}" for strategy in strategies[:1]
            ])
    
    return enhanced_suggestions, total_potential_savings, action_plan