"""
Enhanced debt optimizer that integrates with your existing Flask API
Based on your friend's debt_optimizer_enhanced.py with API compatibility
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Literal, Union, Tuple

# Import your existing config
try:
    from config import DATA_DIRECTORY, OUTPUT_DIRECTORY
except ImportError:
    DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")
    OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "output")

Strategy = Literal["avalanche", "snowball"]
CompoundingMode = Literal["nominal", "effective", "daily"]

@dataclass
class Debt:
    name: str
    balance: float
    apr: float
    min_payment: float
    kind: Optional[str] = None

    def monthly_rate(self, compounding: CompoundingMode = "nominal") -> float:
        r = float(self.apr)
        if compounding == "nominal":
            return r / 12.0
        elif compounding == "effective":
            return (1.0 + r) ** (1.0 / 12.0) - 1.0
        elif compounding == "daily":
            return (1.0 + r / 365.0) ** (365.0 / 12.0) - 1.0
        else:
            return r / 12.0

@dataclass
class BankPayment:
    date: str
    description: str
    amount: float
    debt_name: str
    debt_kind: str

@dataclass
class DebtSummary:
    name: str
    starting_balance: float
    adjusted_balance: float
    apr: float
    min_payment: float
    months: int
    total_paid: float
    interest_paid: float
    bank_payments_found: List[Dict]

DEFAULT_RATES = {"credit_card": 0.22, "personal_loan": 0.16, "overdraft": 0.18, "mortgage": 0.11}
MIN_PAYMENT_HINTS = {"credit_card": {"pct_of_balance": 0.025, "floor": 150.0},
                     "overdraft":   {"pct_of_balance": 0.03,  "floor": 150.0}}

def extract_new_available_income_from_budget_report(report_path: str) -> float:
    """Extract the 'New Available Income' amount from budget report text file."""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for "New Available Income:" pattern
        pattern = r"New Available Income:\*\*\s*R\s*([\d,]+\.?\d*)"
        match = re.search(pattern, content)
        
        if match:
            amount_str = match.group(1).replace(',', '')
            return float(amount_str)
        
        # Fallback patterns
        patterns = [
            r"optimized.*savings.*R\s*([\d,]+\.?\d*)",
            r"total.*potential.*savings.*R\s*([\d,]+\.?\d*)",
            r"improved.*available.*income.*R\s*([\d,]+\.?\d*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                return float(amount_str)
                
        # If no optimized savings found, try current available income
        pattern = r"Available Income:\*\*\s*R\s*([\d,]+\.?\d*)"
        match = re.search(pattern, content)
        if match:
            amount_str = match.group(1).replace(',', '')
            return float(amount_str)
            
        return 0.0
        
    except Exception as e:
        print(f"⚠️  Warning: Could not extract from budget report {report_path}: {e}")
        return 0.0

def discover_budget_report() -> Optional[str]:
    """Find the most recent budget report file with enhanced search."""
    patterns = [
        "enhanced_budget_report_*.txt",
        "budget_report_*.txt", 
        "budget_analysis_*.txt"
    ]
    
    search_paths = [
        os.getcwd(),
        "budget_reports",
        os.path.join(OUTPUT_DIRECTORY, "..", "budget_reports"),
        OUTPUT_DIRECTORY
    ]
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        if os.path.isfile(search_path):
            continue
            
        for pattern in patterns:
            files = glob.glob(os.path.join(search_path, pattern))
            if files:
                found_file = max(files, key=os.path.getmtime)
                return found_file
    
    return None

def extract_debt_payments_from_statement(statement_path: str, debts: List[Debt]) -> Dict[str, List[BankPayment]]:
    """Extract debt payments from bank statement CSV."""
    payments_by_debt: Dict[str, List[BankPayment]] = {debt.name: [] for debt in debts}
    
    # Create mapping for better matching
    debt_kind_map = {debt.kind.lower(): debt.name for debt in debts if debt.kind}
    debt_name_map = {debt.name.lower(): debt.name for debt in debts}
    
    try:
        with open(statement_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                description = str(row.get('Description', '')).strip()
                amount_str = str(row.get('Amount (ZAR)', '') or row.get('Amount', '')).strip()
                date_str = str(row.get('Date', '')).strip()
                
                # Skip if no amount or positive amount (income)
                if not amount_str or amount_str == 'null':
                    continue
                    
                try:
                    amount = float(amount_str.replace(',', ''))
                except (ValueError, TypeError):
                    continue
                    
                if amount >= 0:  # Skip positive amounts (income)
                    continue
                    
                amount = abs(amount)  # Convert to positive for easier handling
                
                # Enhanced pattern matching for debt payments
                debt_payment = extract_debt_from_description(description, debt_kind_map, debt_name_map, debts)
                
                if debt_payment:
                    debt_name, debt_kind = debt_payment
                    payment = BankPayment(
                        date=date_str,
                        description=description,
                        amount=amount,
                        debt_name=debt_name,
                        debt_kind=debt_kind
                    )
                    payments_by_debt[debt_name].append(payment)
                    
    except Exception as e:
        print(f"⚠️  Warning: Could not process bank statement {statement_path}: {e}")
        
    return payments_by_debt

def extract_debt_from_description(description: str, debt_kind_map: Dict[str, str], debt_name_map: Dict[str, str], debts: List[Debt]) -> Optional[Tuple[str, str]]:
    """Enhanced extraction to match debt payment format."""
    desc_lower = description.lower()
    
    # Primary pattern: "Monthly Payment – [Debt Name] ([debt_kind])"
    pattern1 = r'monthly payment\s*[–-]\s*([^(]+)\s*\(([^)]+)\)'
    match = re.search(pattern1, desc_lower)
    if match:
        debt_name_part = match.group(1).strip()
        debt_kind_part = match.group(2).strip()
        
        # Try to match the debt kind first
        if debt_kind_part in debt_kind_map:
            return debt_kind_map[debt_kind_part], debt_kind_part
        
        # Try to match the debt name
        if debt_name_part in debt_name_map:
            return debt_name_map[debt_name_part], debt_kind_part
    
    # Fallback patterns
    for kind, name in debt_kind_map.items():
        if kind in desc_lower and ('payment' in desc_lower or 'monthly' in desc_lower):
            return name, kind
    
    for name_lower, name_actual in debt_name_map.items():
        if name_lower in desc_lower and ('payment' in desc_lower or 'monthly' in desc_lower):
            for debt in debts:
                if debt.name == name_actual:
                    return name_actual, (debt.kind or 'unknown')
    
    return None

def load_debts_from_csv(path: str) -> List[Debt]:
    """Load debts from CSV file."""
    debts: List[Debt] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"name", "balance", "apr"}
        missing = required - set([c.strip().lower() for c in (reader.fieldnames or [])])
        if missing:
            raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            debts.append(
                Debt(
                    name=str(row["name"]).strip(),
                    kind=(str(row.get("kind") or "").strip() or None),
                    balance=float(row["balance"]),
                    apr=float(row["apr"]),
                    min_payment=float(row.get("min_payment") or 0.0),
                )
            )
    if not debts:
        raise ValueError("No debts found in CSV.")
    return debts

def _estimate_min_payment(debt: Debt) -> float:
    """Estimate minimum payment if not provided."""
    if debt.min_payment and debt.min_payment > 0:
        return float(debt.min_payment)
    
    hint = MIN_PAYMENT_HINTS.get((debt.kind or "").lower())
    if hint:
        return float(max(hint["pct_of_balance"] * debt.balance, hint["floor"]))
    
    return float(max(0.015 * debt.balance, 150))

def _order_debts_indices(debts: List[Debt], strategy: Strategy) -> List[int]:
    """Order debts by strategy."""
    if strategy == "avalanche":
        return sorted(range(len(debts)), key=lambda i: (-debts[i].apr, debts[i].balance))
    else:
        return sorted(range(len(debts)), key=lambda i: (debts[i].balance, -debts[i].apr))

def _is_paid_off(x: float, eps: float = 1e-6) -> bool:
    """Check if debt is essentially paid off."""
    return x <= eps

def plan_debt_payoff(
    debts: List[Debt],
    monthly_budget: float,
    strategy: Strategy = "avalanche",
    compounding: CompoundingMode = "nominal",
    max_months: int = 600,
    current_payments: Optional[Dict[str, float]] = None
) -> Dict:
    """Plan debt payoff considering current payments + additional budget."""
    
    # Deep copy debts for simulation
    debts = [Debt(**asdict(d)) for d in debts]
    
    # Initialize tracking variables
    month = 0
    paydown_order: List[str] = []
    per_debt_interest: Dict[str, float] = {d.name: 0.0 for d in debts}
    
    # Calculate total current payments
    current_payments = current_payments or {}
    total_current_payments = sum(current_payments.values())
    total_available_budget = total_current_payments + monthly_budget

    # Main simulation loop
    while month < max_months and any(not _is_paid_off(d.balance) for d in debts):
        active_debts = [d for d in debts if not _is_paid_off(d.balance)]
        
        # 1) Accrue interest
        for d in active_debts:
            interest = d.balance * d.monthly_rate(compounding)
            d.balance += interest
            per_debt_interest[d.name] += interest

        # 2) Pay current minimums first
        remaining_budget = total_available_budget
        for d in active_debts:
            current_payment = current_payments.get(d.name, d.min_payment)
            payment = min(current_payment, d.balance, remaining_budget)
            
            if payment > 0:
                d.balance -= payment
                remaining_budget -= payment

        # 3) Allocate remaining budget using strategy
        while remaining_budget > 1e-8:
            active_debts = [d for d in debts if not _is_paid_off(d.balance)]
            if not active_debts:
                break
                
            order = _order_debts_indices(active_debts, strategy)
            target = active_debts[order[0]]
            
            extra_payment = min(remaining_budget, target.balance)
            target.balance -= extra_payment
            remaining_budget -= extra_payment

        # Track payoff order
        for d in debts:
            if _is_paid_off(d.balance) and d.name not in paydown_order:
                paydown_order.append(d.name)

        month += 1

    # Calculate results
    still_owing = any(d.balance > 1e-6 for d in debts)
    status = "paid_off" if not still_owing else "not_solved_with_current_budget"
    months_to_free = month if status == "paid_off" else None
    total_interest = sum(per_debt_interest.values())

    return {
        "status": status,
        "strategy": strategy,
        "months_to_debt_free": months_to_free,
        "total_interest_paid": round(total_interest, 2),
        "payoff_order": paydown_order,
        "current_payments_total": round(total_current_payments, 2),
        "additional_budget": round(monthly_budget, 2),
    }

def discover_debts_csv(explicit: Optional[str] = None) -> Optional[str]:
    """Auto-discover debts CSV file in common locations."""
    if explicit and os.path.isfile(explicit):
        return explicit
    
    # Check current directory first
    candidates = ["debts.csv", "debts_sample.csv"]
    
    # Check script directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    for candidate in candidates:
        path = os.path.join(script_dir, candidate)
        if os.path.isfile(path):
            return path
    
    # Check data directory
    for candidate in candidates:
        path = os.path.join(DATA_DIRECTORY, candidate)
        if os.path.isfile(path):
            return path
    
    # Check for any CSV with "debt" in the name
    for directory in [script_dir, DATA_DIRECTORY]:
        pattern = os.path.join(directory, "*debt*.csv")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None

def get_enhanced_debt_optimization(available_monthly_budget: float, debts_csv_path: str = None) -> Dict:
    """
    Enhanced debt optimization with budget report integration.
    Compatible with your existing Flask API.
    """
    # Auto-discover debts file if not provided
    if not debts_csv_path:
        debts_csv_path = discover_debts_csv()
        if not debts_csv_path:
            return {
                "error": "No debts CSV file found. Please create a debts.csv file with columns: name, balance, apr, min_payment, kind"
            }
    
    try:
        debts = load_debts_from_csv(debts_csv_path)
        if not debts:
            return {
                "error": "No debts found in CSV file"
            }
        
        # Try to auto-extract budget from report if available
        auto_budget = 0.0
        report_path = discover_budget_report()
        if report_path:
            try:
                auto_budget = extract_new_available_income_from_budget_report(report_path)
                if auto_budget > 0:
                    print(f"📊 Auto-extracted optimized budget: R {auto_budget:,.2f}")
            except Exception as e:
                print(f"⚠️ Could not auto-extract budget: {e}")
        
        # Use the higher of provided budget or auto-extracted budget
        final_budget = max(available_monthly_budget, auto_budget)
        
        # Calculate both strategies
        avalanche_plan = plan_debt_payoff(debts, final_budget, "avalanche")
        snowball_plan = plan_debt_payoff(debts, final_budget, "snowball")
        
        # Calculate interest saved vs minimum payments only
        baseline_plan = plan_debt_payoff(debts, 0.0, "avalanche")
        
        interest_saved_avalanche = None
        interest_saved_snowball = None
        
        if baseline_plan["status"] == "paid_off":
            if avalanche_plan["status"] == "paid_off":
                interest_saved_avalanche = baseline_plan["total_interest_paid"] - avalanche_plan["total_interest_paid"]
            if snowball_plan["status"] == "paid_off":
                interest_saved_snowball = baseline_plan["total_interest_paid"] - snowball_plan["total_interest_paid"]
        
        # Add interest saved to plans
        avalanche_plan["interest_saved_vs_min_only"] = round(interest_saved_avalanche, 2) if interest_saved_avalanche else None
        snowball_plan["interest_saved_vs_min_only"] = round(interest_saved_snowball, 2) if interest_saved_snowball else None
        
        return {
            "avalanche": avalanche_plan,
            "snowball": snowball_plan,
            "recommendation": "avalanche" if avalanche_plan["total_interest_paid"] <= snowball_plan["total_interest_paid"] else "snowball",
            "debts_file_used": debts_csv_path,
            "budget_used": final_budget,
            "auto_extracted_budget": auto_budget if auto_budget > 0 else None
        }
        
    except Exception as e:
        return {
            "error": f"Error processing debts: {str(e)}"
        }