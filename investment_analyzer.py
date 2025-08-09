"""
investment_analyzer.py - Investment portfolio projections
Calculates potential investment returns based on available monthly savings
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

def calculate_investment_projections(monthly_savings: float, years_list: List[int] = None) -> Dict:
    """
    Calculate investment projections for all profiles across different time horizons
    """
    if years_list is None:
        years_list = [1, 5, 10, 15, 20, 25]
    
    if monthly_savings <= 0:
        return {
            "error": "Monthly savings amount must be greater than 0",
            "monthly_savings": monthly_savings
        }
    
    projections = {}
    
    for profile_key, profile in INVESTMENT_PROFILES.items():
        profile_projections = []
        
        for years in years_list:
            # Calculate with both average and effective returns
            avg_future_value = calculate_future_value(monthly_savings, profile.avg_return, years)
            effective_future_value = calculate_future_value(monthly_savings, profile.effective_return, years)
            
            total_contributions = monthly_savings * years * 12
            avg_interest_earned = avg_future_value - total_contributions
            effective_interest_earned = effective_future_value - total_contributions
            
            projection = {
                "years": years,
                "monthly_contribution": round(monthly_savings, 2),
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
        "monthly_savings": monthly_savings,
        "profiles": projections,
        "assumptions": {
            "contribution_frequency": "Monthly (end of month)",
            "compounding_frequency": "Monthly",
            "tax_considerations": "Returns shown are gross (before tax)",
            "inflation_adjustment": "Not included (nominal returns)",
            "fees_included": "No (assumes net-of-fee returns)"
        },
        "recommendations": _generate_investment_recommendations(monthly_savings, projections)
    }

def _generate_investment_recommendations(monthly_savings: float, projections: Dict) -> List[str]:
    """Generate personalized investment recommendations based on savings amount"""
    recommendations = []
    
    # Tax-free savings account recommendation (South African context)
    if monthly_savings <= 2916.67:  # R35,000 annual limit / 12 months
        recommendations.append(
            f"Consider maximizing your Tax-Free Savings Account (TFSA) first - you can invest up to R{monthly_savings * 12:,.0f} per year tax-free"
        )
    else:
        recommendations.append(
            "Consider maxing out your TFSA (R35,000/year) first, then invest the remaining R{:,.0f}/month in other accounts".format(
                monthly_savings - 2916.67
            )
        )
    
    # Portfolio recommendations based on amount
    if monthly_savings < 500:
        recommendations.append(
            "With smaller amounts, consider low-cost index funds or ETFs to minimize fees"
        )
    elif monthly_savings < 2000:
        recommendations.append(
            "You have good investment capacity - consider a diversified portfolio across multiple asset classes"
        )
    else:
        recommendations.append(
            "With substantial monthly savings, consider consulting with a financial advisor for personalized portfolio construction"
        )
    
    # Time horizon recommendations
    conservative_10yr = projections["conservative"]["projections"][2]["effective_future_value"]  # 10 years
    aggressive_10yr = projections["aggressive"]["projections"][2]["effective_future_value"]
    
    difference = aggressive_10yr - conservative_10yr
    recommendations.append(
        f"Over 10 years, aggressive investing could potentially earn R{difference:,.0f} more than conservative, but with higher risk"
    )
    
    return recommendations

def get_investment_analysis(available_monthly_amount: float) -> Dict:
    """
    Main function to get investment analysis
    Returns comprehensive investment projections and recommendations
    """
    try:
        if available_monthly_amount is None or available_monthly_amount < 0:
            return {
                "error": "Invalid monthly amount provided",
                "monthly_amount": available_monthly_amount
            }
        
        if available_monthly_amount == 0:
            return {
                "message": "No funds available for investment",
                "recommendations": [
                    "Focus on building an emergency fund first",
                    "Look for ways to reduce expenses or increase income",
                    "Even small amounts (R50-100/month) can grow significantly over time"
                ],
                "monthly_amount": 0
            }
        
        # Calculate projections
        projections = calculate_investment_projections(available_monthly_amount)
        
        return projections
        
    except Exception as e:
        return {
            "error": f"Error calculating investment projections: {str(e)}",
            "monthly_amount": available_monthly_amount
        }

if __name__ == "__main__":
    # Test with example
    import json
    result = get_investment_analysis(1500)
    print(json.dumps(result, indent=2))