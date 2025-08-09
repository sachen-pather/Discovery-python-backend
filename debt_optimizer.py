#!/usr/bin/env python3
"""
debt_optimizer.py - Debt payoff optimization for the financial analyzer
Integrates with existing budget_analyzer to provide debt payoff strategies
"""
from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Literal

Strategy = Literal["avalanche", "snowball"]

# Import your existing config
try:
    from config import DATA_DIRECTORY, OUTPUT_DIRECTORY
except ImportError:
    DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")
    OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "output")

@dataclass
class Debt:
    name: str
    balance: float
    apr: float
    min_payment: float
    kind: Optional[str] = None

    def monthly_rate(self) -> float:
        return self.apr / 12.0

@dataclass
class PayoffEvent:
    month_index: int
    name: str
    payment: float
    to_interest: float
    to_principal: float
    remaining_balance: float

@dataclass
class DebtSummary:
    name: str
    starting_balance: float
    apr: float
    min_payment: float
    months: int
    total_paid: float
    interest_paid: float

DEFAULT_RATES = {
    "credit_card": 0.22, 
    "personal_loan": 0.16, 
    "overdraft": 0.18, 
    "mortgage": 0.11
}

MIN_PAYMENT_HINTS = {
    "credit_card": {"pct_of_balance": 0.025, "floor": 150},
    "overdraft": {"pct_of_balance": 0.03, "floor": 150}
}

def load_debts_from_csv(path: str) -> List[Debt]:
    """Load debts from CSV file"""
    debts: List[Debt] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
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
    return debts

def discover_debts_csv() -> Optional[str]:
    """Auto-discover debts CSV file in common locations"""
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

def _estimate_min_payment(debt: Debt) -> float:
    """Estimate minimum payment if not provided"""
    if debt.min_payment and debt.min_payment > 0:
        return float(debt.min_payment)
    
    hint = MIN_PAYMENT_HINTS.get((debt.kind or "").lower())
    if hint:
        return float(max(hint["pct_of_balance"] * debt.balance, hint["floor"]))
    
    return float(max(0.015 * debt.balance, 150))

def _order_debts(debts: List[Debt], strategy: Strategy) -> List[int]:
    """Order debts by strategy (avalanche = highest APR first, snowball = lowest balance first)"""
    if strategy == "avalanche":
        return sorted(range(len(debts)), key=lambda i: debts[i].apr, reverse=True)
    else:  # snowball
        return sorted(range(len(debts)), key=lambda i: debts[i].balance)

def _is_paid_off(balance: float, eps: float = 1e-6) -> bool:
    """Check if debt is essentially paid off"""
    return balance <= eps

def plan_debt_payoff(
    debts: List[Debt],
    monthly_budget: float,
    strategy: Strategy = "avalanche",
    max_months: int = 600
) -> Dict:
    """
    Create a debt payoff plan using the specified strategy
    """
    if not debts:
        return {
            "strategy": strategy,
            "months_to_debt_free": 0,
            "total_interest_paid": 0.0,
            "interest_saved_vs_min_only": 0.0,
            "payoff_order": [],
            "debts": [],
            "events": []
        }
    
    # Make copies to avoid modifying originals
    debts = [Debt(**asdict(d)) for d in debts]
    
    # Estimate min payments if not provided
    for d in debts:
        d.min_payment = float(_estimate_min_payment(d))
    
    monthly_budget = max(float(monthly_budget or 0.0), 0.0)
    month = 0
    events: List[PayoffEvent] = []
    debt_histories: Dict[str, List[PayoffEvent]] = {d.name: [] for d in debts}
    
    # Calculate baseline (min payments only)
    def simulate_min_only(active_debts: List[Debt]) -> Dict[str, float]:
        sim = [Debt(**asdict(d)) for d in active_debts]
        months = 0
        total_interest = 0.0
        safety = 2000
        
        while months < safety and any(not _is_paid_off(d.balance) for d in sim):
            # Accrue interest
            for d in sim:
                if _is_paid_off(d.balance): 
                    continue
                interest = d.balance * d.monthly_rate()
                d.balance += interest
                total_interest += interest
            
            # Make minimum payments
            for d in sim:
                if _is_paid_off(d.balance): 
                    continue
                pay = min(d.min_payment, d.balance)
                d.balance -= pay
            
            months += 1
        
        return {"months": months, "interest": total_interest}
    
    baseline = simulate_min_only(debts)
    total_interest = 0.0
    paydown_order: List[str] = []
    
    # Main simulation loop
    while month < max_months and any(not _is_paid_off(d.balance) for d in debts):
        # 1. Accrue interest
        month_interest_total = 0.0
        for d in debts:
            if _is_paid_off(d.balance): 
                continue
            interest = d.balance * d.monthly_rate()
            d.balance += interest
            month_interest_total += interest
        
        total_interest += month_interest_total
        
        # 2. Make minimum payments
        for d in debts:
            if _is_paid_off(d.balance): 
                continue
            pay = min(d.min_payment, d.balance)
            d.balance -= pay
            
            event = PayoffEvent(
                month_index=month,
                name=d.name,
                payment=pay,
                to_interest=0.0,
                to_principal=pay,
                remaining_balance=max(d.balance, 0.0)
            )
            debt_histories[d.name].append(event)
            events.append(event)
        
        # 3. Allocate extra budget using strategy
        order = _order_debts(debts, strategy)
        leftover = monthly_budget
        
        for idx in order:
            d = debts[idx]
            if _is_paid_off(d.balance) or leftover <= 0: 
                continue
            
            pay = min(leftover, d.balance)
            d.balance -= pay
            leftover -= pay
            
            event = PayoffEvent(
                month_index=month,
                name=d.name,
                payment=pay,
                to_interest=0.0,
                to_principal=pay,
                remaining_balance=max(d.balance, 0.0)
            )
            debt_histories[d.name].append(event)
            events.append(event)
        
        # Track when debts are paid off
        for d in debts:
            if _is_paid_off(d.balance) and d.name not in paydown_order:
                paydown_order.append(d.name)
        
        month += 1
        
        # Safety check for infinite loops
        if month > 1 and month % 120 == 0:
            if all(d.min_payment <= d.balance * d.monthly_rate() + 1e-2 
                   for d in debts if not _is_paid_off(d.balance)):
                break
    
    months_to_free = month
    
    # Create debt summaries
    summaries: List[DebtSummary] = []
    for d in debts:
        hist = debt_histories[d.name]
        if hist:
            total_paid = sum(e.payment for e in hist)
            months_spanned = 1 + (max(e.month_index for e in hist) - min(e.month_index for e in hist))
            starting_bal_guess = hist[0].remaining_balance + hist[0].payment
        else:
            total_paid = 0.0
            months_spanned = 0
            starting_bal_guess = 0.0
        
        summaries.append(DebtSummary(
            name=d.name,
            starting_balance=starting_bal_guess,
            apr=d.apr,
            min_payment=d.min_payment,
            months=months_spanned,
            total_paid=total_paid,
            interest_paid=0.0  # Will be calculated below
        ))
    
    # Approximate interest allocation per debt
    avg_balance_est: Dict[str, float] = {}
    for name, entries in debt_histories.items():
        if not entries:
            avg_balance_est[name] = 0.0
        else:
            avg_balance_est[name] = sum(e.remaining_balance for e in entries) / max(len(entries), 1)
    
    total_avg = sum(avg_balance_est.values()) or 1.0
    for s in summaries:
        share = (avg_balance_est.get(s.name, 0.0) / total_avg)
        s.interest_paid = share * total_interest
    
    interest_saved = max(baseline["interest"] - total_interest, 0.0)
    
    return {
        "strategy": strategy,
        "months_to_debt_free": months_to_free,
        "total_interest_paid": round(total_interest, 2),
        "interest_saved_vs_min_only": round(interest_saved, 2),
        "payoff_order": paydown_order,
        "debts": [asdict(s) for s in summaries],
        "events": [asdict(e) for e in events]
    }

def get_debt_optimization(available_monthly_budget: float, debts_csv_path: str = None) -> Dict:
    """
    Main function to get debt optimization plan
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
        
        # Calculate both strategies
        avalanche_plan = plan_debt_payoff(debts, available_monthly_budget, "avalanche")
        snowball_plan = plan_debt_payoff(debts, available_monthly_budget, "snowball")
        
        return {
            "avalanche": avalanche_plan,
            "snowball": snowball_plan,
            "recommendation": "avalanche" if avalanche_plan["total_interest_paid"] <= snowball_plan["total_interest_paid"] else "snowball",
            "debts_file_used": debts_csv_path
        }
        
    except Exception as e:
        return {
            "error": f"Error processing debts: {str(e)}"
        }

if __name__ == "__main__":
    # Test with example usage
    result = get_debt_optimization(1000)
    print(json.dumps(result, indent=2))