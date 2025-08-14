"""
investment_analyzer.py - UPDATED VERSION with debt/investment split support
Calculates potential investment returns based on allocated portion of available monthly savings
"""

from dataclasses import dataclass
from typing import Dict, List
import math

@dataclass
class InvestmentProfile:
    name: str
    avg_return: float     # Average expected annual return (decimal)
    volatility: float     # Annual volatility (decimal)
    description: str

    @property
    def effective_return(self) -> float:
        """Volatility-adjusted return using: effective ≈ avg − 0.5·sigma²"""
        return max(0.0, self.avg_return - 0.5 * (self.volatility ** 2))

# South African investment profiles
INVESTMENT_PROFILES = {
    "conservative": InvestmentProfile(
        name="Conservative",
        avg_return=0.065,  # 6.5%
        volatility=0.06,   # 6%
        description="Low-risk portfolio focused on bonds, money market funds, and stable investments"
    ),
    "moderate": InvestmentProfile(
        name="Moderate", 
        avg_return=0.085,  # 8.5%
        volatility=0.12,   # 12%
        description="Balanced portfolio with mix of equities, bonds, and alternative investments"
    ),
    "aggressive": InvestmentProfile(
        name="Aggressive",
        avg_return=0.105,  # 10.5%
        volatility=0.18,   # 18%
        description="High-risk, high-reward portfolio focused on equities and growth investments"
    )
}

def calculate_future_value(monthly_contribution: float, annual_rate: float, years: int) -> float:
    """
    Calculate future value of monthly contributions with compound interest
    Uses the future value of ordinary annuity formula
    """
    if annual_rate == 0 or years == 0 or monthly_contribution == 0:
        return monthly_contribution * years * 12
    
    # Convert to monthly rate
    monthly_rate = (1 + annual_rate) ** (1/12) - 1
    total_months = years * 12
    
    # Future value of ordinary annuity formula
    future_value = monthly_contribution * (((1 + monthly_rate) ** total_months - 1) / monthly_rate)
    
    return future_value

def calculate_investment_projections(investment_budget: float, years_list: List[int] = None) -> Dict:
    """
    Calculate investment projections for all profiles across different time horizons
    UPDATED: Now uses investment_budget parameter instead of monthly_savings
    """
    if years_list is None:
        years_list = [1, 5, 10, 15, 20, 25]
    
    if investment_budget <= 0:
        return {
            "error": "Investment budget must be greater than 0",
            "investment_budget": investment_budget
        }
    
    projections = {}
    
    for profile_key, profile in INVESTMENT_PROFILES.items():
        profile_projections = []
        
        for years in years_list:
            # Calculate with both average and effective returns
            avg_future_value = calculate_future_value(investment_budget, profile.avg_return, years)
            effective_future_value = calculate_future_value(investment_budget, profile.effective_return, years)
            
            total_contributions = investment_budget * years * 12
            avg_interest_earned = avg_future_value - total_contributions
            effective_interest_earned = effective_future_value - total_contributions
            
            projection = {
                "years": years,
                "monthly_contribution": round(investment_budget, 2),
                "total_contributions": round(total_contributions, 2),
                "avg_annual_return": round(profile.avg_return * 100, 2),
                "volatility": round(profile.volatility * 100, 2),
                "effective_annual_return": round(profile.effective_return * 100, 2),
                "avg_future_value": round(avg_future_value, 2),
                "effective_future_value": round(effective_future_value, 2),
                "avg_interest_earned": round(avg_interest_earned, 2),
                "effective_interest_earned": round(effective_interest_earned, 2),
                "avg_roi_percentage": round((avg_interest_earned / total_contributions * 100) if total_contributions > 0 else 0, 2),
                "effective_roi_percentage": round((effective_interest_earned / total_contributions * 100) if total_contributions > 0 else 0, 2)
            }
            
            profile_projections.append(projection)
        
        projections[profile_key] = {
            "profile": {
                "name": profile.name,
                "description": profile.description,
                "avg_return": round(profile.avg_return * 100, 2),
                "volatility": round(profile.volatility * 100, 2),
                "effective_return": round(profile.effective_return * 100, 2)
            },
            "projections": profile_projections
        }
    
    return {
        "investment_budget": investment_budget,
        "profiles": projections,
        "assumptions": {
            "contribution_frequency": "Monthly (end of month)",
            "compounding_frequency": "Monthly",
            "tax_considerations": "Returns shown are gross (before tax)",
            "inflation_adjustment": "Not included (nominal returns)",
            "fees_included": "No (assumes net-of-fee returns)"
        },
        "recommendations": _generate_investment_recommendations(investment_budget, projections)
    }

def _generate_investment_recommendations(investment_budget: float, projections: Dict) -> List[str]:
    """Generate personalized investment recommendations based on investment budget amount"""
    recommendations = []
    
    # Tax-free savings account recommendation (South African context)
    if investment_budget <= 2916.67:  # R35,000 annual limit / 12 months
        recommendations.append(
            f"Consider maximizing your Tax-Free Savings Account (TFSA) first - you can invest up to R{investment_budget * 12:,.0f} per year tax-free"
        )
    else:
        recommendations.append(
            "Consider maxing out your TFSA (R35,000/year) first, then invest the remaining R{:,.0f}/month in other accounts".format(
                investment_budget - 2916.67
            )
        )
    
    # Portfolio recommendations based on amount
    if investment_budget < 500:
        recommendations.append(
            "With smaller amounts, consider low-cost index funds or ETFs to minimize fees"
        )
    elif investment_budget < 2000:
        recommendations.append(
            "You have good investment capacity - consider a diversified portfolio across multiple asset classes"
        )
    else:
        recommendations.append(
            "With substantial monthly investment budget, consider consulting with a financial advisor for personalized portfolio construction"
        )
    
    # Time horizon recommendations
    conservative_10yr = projections["conservative"]["projections"][2]["effective_future_value"]  # 10 years
    aggressive_10yr = projections["aggressive"]["projections"][2]["effective_future_value"]
    
    difference = aggressive_10yr - conservative_10yr
    recommendations.append(
        f"Over 10 years, aggressive investing could potentially earn R{difference:,.0f} more than conservative, but with higher risk"
    )
    
    return recommendations

def get_investment_analysis(
    total_available_income: float,      # UPDATED: Total available income
    investment_allocation_ratio: float  # NEW: 0.0 to 1.0 (e.g., 0.3 = 30% to investment)
) -> Dict:
    """
    Main function to get investment analysis with debt/investment split support
    UPDATED VERSION - now uses allocation ratios instead of absolute amounts
    """
    try:
        # Calculate actual investment budget from ratio
        investment_budget = total_available_income * investment_allocation_ratio
        
        print(f"💰 Total available income: R{total_available_income:.2f}")
        print(f"📊 Investment allocation ratio: {investment_allocation_ratio*100:.1f}%")
        print(f"📈 Investment budget: R{investment_budget:.2f}")
        
        if total_available_income is None or total_available_income < 0:
            return {
                "error": "Invalid total available income provided",
                "total_available_income": total_available_income,
                "investment_allocation_ratio": investment_allocation_ratio
            }
        
        if investment_allocation_ratio < 0 or investment_allocation_ratio > 1:
            return {
                "error": "Investment allocation ratio must be between 0.0 and 1.0",
                "total_available_income": total_available_income,
                "investment_allocation_ratio": investment_allocation_ratio
            }
        
        if investment_budget == 0:
            return {
                "message": "No funds allocated for investment",
                "recommendations": [
                    "Consider allocating some budget to investment for long-term growth",
                    "Even small amounts (R50-100/month) can grow significantly over time",
                    "Review your debt/investment allocation strategy"
                ],
                "total_available_income": total_available_income,
                "investment_allocation_ratio": investment_allocation_ratio,
                "investment_budget": 0
            }
        
        # Calculate projections using the allocated investment budget
        projections = calculate_investment_projections(investment_budget)
        
        # Add allocation information to the result
        projections.update({
            "total_available_income": total_available_income,
            "investment_allocation_ratio": investment_allocation_ratio,
            "allocation_info": {
                "percentage_to_investment": round(investment_allocation_ratio * 100, 1),
                "monthly_amount_invested": round(investment_budget, 2),
                "annual_amount_invested": round(investment_budget * 12, 2)
            }
        })
        
        return projections
        
    except Exception as e:
        return {
            "error": f"Error calculating investment projections: {str(e)}",
            "total_available_income": total_available_income,
            "investment_allocation_ratio": investment_allocation_ratio
        }

# For backwards compatibility
def get_investment_analysis_legacy(available_monthly_amount: float) -> Dict:
    """Legacy function for backwards compatibility - assumes 100% allocation to investment."""
    return get_investment_analysis(
        total_available_income=available_monthly_amount,
        investment_allocation_ratio=1.0  # 100% to investment
    )

if __name__ == "__main__":
    # Test with example
    import json
    result = get_investment_analysis(
        total_available_income=2000,
        investment_allocation_ratio=0.3  # 30% to investment = R600
    )
    print(json.dumps(result, indent=2))