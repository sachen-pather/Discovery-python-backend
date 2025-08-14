"""
Enhanced debt optimizer - UPDATED VERSION with debt/investment split support
Properly detects current payments from bank statement data and uses allocation ratios
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
    current_payment: Optional[float] = None  # Detected from bank statement

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

def extract_current_payments_from_categorized_file(categorized_file_path: str, debts: List[Debt]) -> Dict[str, float]:
    """
    Extract current debt payments from categorized CSV file.
    This is the FIXED version that looks at the actual categorized file!
    """
    current_payments = {}
    
    if not os.path.exists(categorized_file_path):
        print(f"❌ Categorized file not found: {categorized_file_path}")
        return current_payments
    
    try:
        import pandas as pd
        df = pd.read_csv(categorized_file_path)
        print(f"🔍 Looking for debt payments in categorized file with {len(df)} transactions")
        
        # Create mapping for matching
        debt_name_map = {}
        debt_kind_map = {}
        
        for debt in debts:
            # Map by name (case insensitive)
            debt_name_map[debt.name.lower()] = debt.name
            # Map by kind
            if debt.kind:
                debt_kind_map[debt.kind.lower()] = debt.name
        
        print(f"🗂️ Debt mappings - Names: {debt_name_map}, Kinds: {debt_kind_map}")
        
        # Look for debt payments in the categorized transactions
        for _, transaction in df.iterrows():
            # Check if this is categorized as debt payment
            category = str(transaction.get('Category', '')).lower()
            description = str(transaction.get('Description', '')).lower()
            
            # Skip if not a debt payment
            if 'debt' not in category and 'payment' not in category:
                continue
                
            # Extract payment amount (should be positive for analysis)
            amount = abs(float(transaction.get('Amount (ZAR)', 0) or transaction.get('Amount', 0) or 0))
            if amount <= 0:
                continue
            
            print(f"💳 Found potential debt payment: {description} = R{amount}")
            
            # Try to match to a specific debt
            debt_name = None
            
            # Method 1: Look for debt kind in parentheses - "Monthly Payment – Mortgage (mortgage)"
            kind_match = re.search(r'\(([^)]+)\)', description)
            if kind_match:
                kind_in_desc = kind_match.group(1).lower()
                if kind_in_desc in debt_kind_map:
                    debt_name = debt_kind_map[kind_in_desc]
                    print(f"✅ Matched by kind in parentheses: {kind_in_desc} -> {debt_name}")
            
            # Method 2: Look for debt name in description
            if not debt_name:
                for name_key, name_value in debt_name_map.items():
                    if name_key in description:
                        debt_name = name_value
                        print(f"✅ Matched by name: {name_key} -> {debt_name}")
                        break
            
            # Method 3: Look for debt kind anywhere in description
            if not debt_name:
                for kind_key, kind_value in debt_kind_map.items():
                    if kind_key in description:
                        debt_name = kind_value
                        print(f"✅ Matched by kind: {kind_key} -> {debt_name}")
                        break
            
            # Add to current payments
            if debt_name:
                current_payments[debt_name] = current_payments.get(debt_name, 0) + amount
                print(f"💰 Added payment: {debt_name} = R{amount} (total: R{current_payments[debt_name]})")
            else:
                print(f"⚠️ Could not match debt payment: {description}")
        
        print(f"✅ Final detected current payments: {current_payments}")
        return current_payments
        
    except Exception as e:
        print(f"❌ Error extracting current payments: {e}")
        import traceback
        traceback.print_exc()
        return {}

def _estimate_min_payment(debt: Debt) -> float:
    """Estimate minimum payment if not provided."""
    if debt.min_payment and debt.min_payment > 0:
        return float(debt.min_payment)
    
    # Use current payment as minimum if available
    if debt.current_payment and debt.current_payment > 0:
        return float(debt.current_payment)
    
    # Fallback to percentage of balance
    return float(max(0.02 * debt.balance, 150))

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
    additional_budget: float,
    strategy: Strategy = "avalanche",
    compounding: CompoundingMode = "nominal",
    max_months: int = 600,
    current_payments: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Plan debt payoff considering current payments + additional budget.
    FIXED VERSION - properly calculates with existing payments.
    """
    
    # Deep copy debts for simulation
    debts = [Debt(**asdict(d)) for d in debts]
    current_payments = current_payments or {}
    
    # Set current payments on debts and ensure minimum payments
    total_current_payments = 0
    for debt in debts:
        detected_payment = current_payments.get(debt.name, 0)
        if detected_payment > 0:
            debt.current_payment = detected_payment
            total_current_payments += detected_payment
            # Use detected payment as minimum if higher than stated minimum
            debt.min_payment = max(debt.min_payment, detected_payment)
        else:
            debt.min_payment = _estimate_min_payment(debt)
    
    total_budget = total_current_payments + additional_budget
    
    print(f"💰 Current payments detected: R{total_current_payments:.2f}")
    print(f"💰 Additional budget available: R{additional_budget:.2f}")
    print(f"💰 Total payment capacity: R{total_budget:.2f}")
    
    # Check if we can even cover minimum payments
    total_minimums = sum(debt.min_payment for debt in debts)
    if total_budget < total_minimums:
        return {
            "status": "insufficient_budget",
            "strategy": strategy,
            "months_to_debt_free": None,
            "total_interest_paid": 999999999.99,  # Large number instead of infinity
            "payoff_order": [],
            "current_payments_total": round(total_current_payments, 2),
            "additional_budget": round(additional_budget, 2),
            "total_minimums_required": round(total_minimums, 2),
            "budget_shortfall": round(total_minimums - total_budget, 2)
        }
    
    # Initialize tracking variables
    month = 0
    paydown_order: List[str] = []
    per_debt_interest: Dict[str, float] = {d.name: 0.0 for d in debts}
    
    # Simulate minimum payments only for comparison
    def simulate_minimum_only():
        sim_debts = [Debt(**asdict(d)) for d in debts]
        sim_months = 0
        sim_interest = 0.0
        
        while sim_months < max_months and any(not _is_paid_off(d.balance) for d in sim_debts):
            # Accrue interest
            for d in sim_debts:
                if not _is_paid_off(d.balance):
                    interest = d.balance * d.monthly_rate(compounding)
                    d.balance += interest
                    sim_interest += interest
            
            # Make minimum payments only
            for d in sim_debts:
                if not _is_paid_off(d.balance):
                    payment = min(d.min_payment, d.balance)
                    d.balance -= payment
            
            sim_months += 1
            
            # Safety break for infinite loops
            if sim_months > 0 and sim_months % 120 == 0:
                avg_interest_vs_payment = sim_interest / max(sim_months * sum(d.min_payment for d in debts), 1)
                if avg_interest_vs_payment > 0.8:  # Interest is eating most of the payment
                    break
        
        return sim_months, sim_interest

    baseline_months, baseline_interest = simulate_minimum_only()

    # Main simulation loop with optimized payments
    while month < max_months and any(not _is_paid_off(d.balance) for d in debts):
        active_debts = [d for d in debts if not _is_paid_off(d.balance)]
        
        # 1) Accrue interest
        for d in active_debts:
            interest = d.balance * d.monthly_rate(compounding)
            d.balance += interest
            per_debt_interest[d.name] += interest

        # 2) Make minimum payments
        remaining_budget = total_budget
        for d in active_debts:
            payment = min(d.min_payment, d.balance, remaining_budget)
            if payment > 0:
                d.balance -= payment
                remaining_budget -= payment

        # 3) Allocate extra budget using strategy
        while remaining_budget > 1e-8:
            active_debts = [d for d in debts if not _is_paid_off(d.balance)]
            if not active_debts:
                break
                
            order = _order_debts_indices(active_debts, strategy)
            target = active_debts[order[0]]
            
            extra_payment = min(remaining_budget, target.balance)
            if extra_payment <= 1e-8:
                break
                
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
    
    # Calculate interest saved
    interest_saved = None
    if status == "paid_off" and baseline_months < max_months:
        interest_saved = max(0, baseline_interest - total_interest)

    return {
        "status": status,
        "strategy": strategy,
        "months_to_debt_free": months_to_free,
        "total_interest_paid": round(total_interest, 2),
        "interest_saved_vs_min_only": round(interest_saved, 2) if interest_saved is not None else None,
        "payoff_order": paydown_order,
        "current_payments_total": round(total_current_payments, 2),
        "additional_budget": round(additional_budget, 2),
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
    
    return None

def make_json_safe(obj):
    """Convert any infinity or NaN values to JSON-safe equivalents."""
    import math
    
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, float):
        if math.isinf(obj):
            return 999999999.99 if obj > 0 else -999999999.99
        elif math.isnan(obj):
            return 0.0
        else:
            return obj
    else:
        return obj

def get_enhanced_debt_optimization(
    total_available_income: float,      # UPDATED: Total available income
    debt_allocation_ratio: float,       # NEW: 0.0 to 1.0 (e.g., 0.8 = 80% to debt)
    debts_csv_path: str = None,
    categorized_file_path: str = None
) -> Dict:
    """
    Enhanced debt optimization with debt/investment split support.
    UPDATED VERSION - now uses allocation ratios instead of absolute amounts.
    """
    # Calculate actual debt payment budget from ratio
    debt_payment_budget = total_available_income * debt_allocation_ratio
    
    print(f"💰 Total available income: R{total_available_income:.2f}")
    print(f"📊 Debt allocation ratio: {debt_allocation_ratio*100:.1f}%")
    print(f"💳 Debt payment budget: R{debt_payment_budget:.2f}")
    
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
        
        # Extract current payments from categorized file
        current_payments = {}
        if categorized_file_path and os.path.exists(categorized_file_path):
            print(f"🔍 Looking for current payments in: {categorized_file_path}")
            current_payments = extract_current_payments_from_categorized_file(categorized_file_path, debts)
        else:
            print(f"⚠️ No categorized file provided or found: {categorized_file_path}")
        
        # Calculate both strategies using the allocated debt budget
        avalanche_plan = plan_debt_payoff(
            debts, 
            debt_payment_budget,  # UPDATED: Use allocated budget
            "avalanche",
            current_payments=current_payments
        )
        snowball_plan = plan_debt_payoff(
            debts, 
            debt_payment_budget,  # UPDATED: Use allocated budget
            "snowball",
            current_payments=current_payments
        )
        
        # Determine recommendation
        recommendation = "avalanche"
        if (avalanche_plan["status"] == "paid_off" and snowball_plan["status"] == "paid_off"):
            if snowball_plan["total_interest_paid"] < avalanche_plan["total_interest_paid"]:
                recommendation = "snowball"
        elif snowball_plan["status"] == "paid_off" and avalanche_plan["status"] != "paid_off":
            recommendation = "snowball"
        
        result = {
            "avalanche": make_json_safe(avalanche_plan),
            "snowball": make_json_safe(snowball_plan),
            "recommendation": recommendation,
            "debts_file_used": debts_csv_path,
            "categorized_file_used": categorized_file_path,
            "total_available_income": total_available_income,  # NEW
            "debt_allocation_ratio": debt_allocation_ratio,    # NEW
            "debt_budget_used": debt_payment_budget,          # NEW
            "current_payments_detected": current_payments
        }
        
        return make_json_safe(result)
        
    except Exception as e:
        print(f"❌ Error in debt optimization: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Error processing debts: {str(e)}"
        }

# For backwards compatibility
def get_debt_optimization(available_monthly_budget: float, debts_csv_path: str = None) -> Dict:
    """Backwards compatible function - assumes 100% allocation to debt."""
    return get_enhanced_debt_optimization(
        total_available_income=available_monthly_budget,
        debt_allocation_ratio=1.0,  # 100% to debt
        debts_csv_path=debts_csv_path
    )