# llm_categorizer.py
"""
LLM-powered bank transaction categorizer (OpenAI or Groq) with safe fallbacks + hard overrides.

Usage:
    from llm_categorizer import categorize_with_llm
    df_out = categorize_with_llm(
        df,
        desc_col="Description",
        amount_col="Amount (ZAR)",
        date_col="Date",
        provider="openai",   # or "groq" or "auto"
        batch_size=25,
        max_retries=2
    )

Environment:
    export OPENAI_API_KEY="..."   # optional
    export GROQ_API_KEY="..."     # optional
"""

import os
import re
import json
import time
import math
from typing import List, Dict, Any, Optional

import pandas as pd

# ------------------ Allowed labels ------------------

ALLOWED_CATEGORIES = [
    "Income",
    "Debt Payments",
    "Rent/Mortgage",
    "Subscriptions",
    "Dining Out",
    "Transport",
    "Groceries",
    "Shopping",
    "Administrative",
    "Other",
]

ALLOWED_DEBT_KINDS = [
    "mortgage",
    "credit_card",
    "personal_loan",
    "store_card",
    "auto_loan",
    "student_loan",
]

# Parenthetical kind extractor: "… (credit_card)"
PAREN_KIND_RE = re.compile(r"\(([^()]+)\)\s*$")

def _extract_parenthetical_kind(desc: str) -> Optional[str]:
    if not isinstance(desc, str):
        return None
    m = PAREN_KIND_RE.search(desc.strip())
    if not m:
        return None
    kind = m.group(1).strip().lower().replace(" ", "_")
    return kind

# ------------------ Rule-based fallback ------------------

def _fallback_rule_category(description: str, amount: Optional[float]) -> Dict[str, Any]:
    dl = (description or "").lower()

    # Income (by keywords or positive amount)
    if any(w in dl for w in ["salary", "wage", "overtime", "pay", "payout", "deposit"]) or (
        amount is not None and not pd.isna(amount) and float(amount) > 0
    ):
        return {"Category": "Income", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}

    # Administrative
    if any(w in dl for w in ["opening balance", "account maintenance", "atm fee", "bank fee"]):
        return {"Category": "Administrative", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}

    # Debt detection via trigger words or parenthetical hint
    debt_trigs = ["monthly payment", "repayment", "installment", "instalment", "debit order", "debit-order", "debitorder"]
    if any(t in dl for t in debt_trigs) or _extract_parenthetical_kind(description):
        debt_kind = _extract_parenthetical_kind(description)
        debt_name = PAREN_KIND_RE.sub("", description or "").strip()
        if debt_kind not in ALLOWED_DEBT_KINDS:
            debt_kind = None
        return {"Category": "Debt Payments", "IsDebtPayment": True, "DebtKind": debt_kind, "DebtName": debt_name}

    # Housing (non-debt)
    if any(w in dl for w in ["rent", "room share", "accommodation"]) or "mortgage" in dl:
        return {"Category": "Rent/Mortgage", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}

    # Groceries
    if any(w in dl for w in ["shoprite", "pick n pay", "checkers", "spar", "woolworths", "groceries", "food store"]):
        return {"Category": "Groceries", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}

    # Dining Out
    if any(w in dl for w in ["kfc", "mcdonald", "restaurant", "takeaway", "pizza", "burger", "nando"]):
        return {"Category": "Dining Out", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}

    # Transport
    if any(w in dl for w in ["taxi", "uber", "bolt", "transport", "fuel", "petrol", "bus fare"]):
        return {"Category": "Transport", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}

    # Subscriptions & utilities
    if any(w in dl for w in ["insurance", "subscription", "netflix", "dstv", "vodacom", "mtn", "airtime", "data", "electricity", "prepaid electricity", "water", "municipal"]):
        return {"Category": "Subscriptions", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}

    # Shopping
    if any(w in dl for w in ["pep stores", "clothing", "edgars", "truworths", "shopping", "clothes", "fashion"]):
        return {"Category": "Shopping", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}

    return {"Category": "Other", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}

# ------------------ Prompting ------------------

SYSTEM_PROMPT = (
    "You are a precise bank-transaction categorizer for South African statements. "
    "Return STRICT JSON with one object per input row. Allowed Category values: "
    + ", ".join(f'\"{c}\"' for c in ALLOWED_CATEGORIES) + ". "
    "A transaction is a DEBT PAYMENT if it is a repayment/installment to a credit product "
    "(mortgage/home loan, credit card, personal loan, store card, auto/vehicle finance, student loan or similar from description). "
    "If it is a debt payment, set Category to \"Debt Payments\", IsDebtPayment=true, "
    "DebtKind to one of "
    + ", ".join(f'\"{k}\"' for k in ALLOWED_DEBT_KINDS)
    + ", otherwise null. "
    "Also return DebtName as a human-readable label (often description without the parenthetical kind). "
    "Income (salary/wages) should be Category=\"Income\" with IsDebtPayment=false.\n\n"
    "Output schema exactly:\n"
    "{ \"results\": [ {"
    "\"Category\": str, "
    "\"IsDebtPayment\": bool, "
    "\"DebtKind\": str|null, "
    "\"DebtName\": str|null } ... ] }"
)

def _make_user_prompt(rows: List[Dict[str, Any]]) -> str:
    lines = ["Categorize the following transactions:\n"]
    for r in rows:
        amt = "null" if r["amount"] is None or (isinstance(r["amount"], float) and math.isnan(r["amount"])) else r["amount"]
        lines.append(f"- #{r['index']} | {r['date']} | {r['description']} | amount={amt}")
    lines.append("\nReturn JSON for the SAME order of #indices.")
    return "\n".join(lines)

# ------------------ Hard overrides (ALWAYS-win rules) ------------------

REFUND_WORDS = ["refund", "reversal", "cashback", "chargeback", "reimburs"]
GROCERY_WORDS = ["shoprite", "pick n pay", "checkers", "spar", "woolworths", "grocer", "food store"]
DINING_WORDS = ["kfc", "mcdonald", "restaurant", "takeaway", "pizza", "burger", "nando"]
TRANSPORT_WORDS = ["taxi", "uber", "bolt", "transport", "fuel", "petrol", "bus fare"]
DEBT_TRIGS = ["monthly payment", "repayment", "installment", "instalment", "debit order", "debit-order", "debitorder", "loan payment"]

def _looks_like_debt(desc_lower: str) -> bool:
    return any(t in desc_lower for t in DEBT_TRIGS) or _extract_parenthetical_kind(desc_lower) is not None

def _force_rules(desc: str, amt: Optional[float]) -> Optional[dict]:
    """Return a forced correction dict or None to accept LLM output."""
    dl = (desc or "").lower()

    # 1) Explicit debt
    if _looks_like_debt(dl) or "mortgage" in dl:
        kind = _extract_parenthetical_kind(desc)
        if kind not in ALLOWED_DEBT_KINDS:
            # salvage common keywords
            if "credit card" in dl: kind = "credit_card"
            elif "personal loan" in dl: kind = "personal_loan"
            elif "store card" in dl: kind = "store_card"
            elif "vehicle" in dl or "auto" in dl or "car finance" in dl: kind = "auto_loan"
            elif "student" in dl or "education" in dl: kind = "student_loan"
            elif "mortgage" in dl or "home loan" in dl or "bond" in dl: kind = "mortgage"
            else: kind = None
        name = PAREN_KIND_RE.sub("", desc or "").strip() or None
        return {"Category": "Debt Payments", "IsDebtPayment": True, "DebtKind": kind, "DebtName": name}

    # 2) Sign sanity + merchant hints
    if amt is not None and not pd.isna(amt):
        if float(amt) < 0 and not any(w in dl for w in REFUND_WORDS):
            if any(w in dl for w in GROCERY_WORDS):
                return {"Category": "Groceries", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}
            if any(w in dl for w in DINING_WORDS):
                return {"Category": "Dining Out", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}
            if any(w in dl for w in TRANSPORT_WORDS):
                return {"Category": "Transport", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}
            if "rent" in dl or "accommodation" in dl:
                return {"Category": "Rent/Mortgage", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}
        if float(amt) > 0 and not any(w in dl for w in REFUND_WORDS):
            if any(w in dl for w in ["salary", "wage", "overtime", "pay", "payout", "deposit"]):
                return {"Category": "Income", "IsDebtPayment": False, "DebtKind": None, "DebtName": None}

    return None

# ------------------ Providers ------------------

def _openai_categorize(batch_rows: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = _make_user_prompt(batch_rows)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        if not (isinstance(data, dict) and "results" in data and isinstance(data["results"], list)):
            return None
        return data["results"]
    except Exception:
        return None

def _groq_categorize(batch_rows: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = _make_user_prompt(batch_rows)
        resp = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\nReturn ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content.strip()
        start = content.find("{"); end = content.rfind("}")
        if start == -1 or end == -1:
            return None
        content = content[start:end + 1]
        data = json.loads(content)
        if not (isinstance(data, dict) and "results" in data and isinstance(data["results"], list)):
            return None
        return data["results"]
    except Exception:
        return None

# ------------------ Public API ------------------

def categorize_with_llm(
    df: pd.DataFrame,
    desc_col: str,
    amount_col: Optional[str],
    date_col: Optional[str],
    provider: str = "openai",
    batch_size: int = 30,
    max_retries: int = 2
) -> pd.DataFrame:
    """
    Adds/overwrites columns: Category, IsDebtPayment, DebtKind, DebtName.
    provider: "openai" | "groq" | "auto"
    """
    out = df.copy()

    # Ensure output columns exist
    for col in ["Category", "IsDebtPayment", "DebtKind", "DebtName"]:
        if col not in out.columns:
            out[col] = None

    # Normalize amount to numeric
    if amount_col and amount_col in out.columns:
        out[amount_col] = pd.to_numeric(out[amount_col], errors="coerce")
    else:
        amount_col = None

    n = len(out)
    indices = list(range(n))

    def _do_request(rows: List[Dict[str, Any]], which: str) -> Optional[List[Dict[str, Any]]]:
        if which == "openai":
            return _openai_categorize(rows)
        if which == "groq":
            return _groq_categorize(rows)
        return None

    for i in range(0, n, batch_size):
        batch_idx = indices[i:i + batch_size]
        rows = []
        for j in batch_idx:
            rows.append({
                "index": j,
                "date": str(out.iloc[j][date_col]) if date_col and date_col in out.columns else "",
                "description": str(out.iloc[j][desc_col]),
                "amount": float(out.iloc[j][amount_col]) if amount_col else None,
            })

        # Determine provider each attempt
        results = None
        for attempt in range(max_retries + 1):
            chosen = provider
            if provider == "auto":
                chosen = "openai" if os.getenv("OPENAI_API_KEY") else "groq"

            results = _do_request(rows, chosen)
            if results is not None and len(results) == len(rows):
                break

            if provider == "auto" and attempt == 0:
                alt = "groq" if chosen == "openai" else "openai"
                results = _do_request(rows, alt)
                if results is not None and len(results) == len(rows):
                    break

            time.sleep(0.6 * (attempt + 1))

        # Fill from LLM or fallback
        for k, r in enumerate(rows):
            j = r["index"]
            item = None
            if results and k < len(results):
                item = results[k]

            # Start with fallback defaults
            if not item or not isinstance(item, dict):
                forced = _force_rules(r["description"], r["amount"])
                if forced:
                    out.at[j, "Category"] = forced["Category"]
                    out.at[j, "IsDebtPayment"] = forced["IsDebtPayment"]
                    out.at[j, "DebtKind"] = forced["DebtKind"]
                    out.at[j, "DebtName"] = forced["DebtName"]
                    continue
                fb = _fallback_rule_category(r["description"], r["amount"])
                out.at[j, "Category"] = fb["Category"]
                out.at[j, "IsDebtPayment"] = fb["IsDebtPayment"]
                out.at[j, "DebtKind"] = fb["DebtKind"]
                out.at[j, "DebtName"] = fb["DebtName"]
                continue

            # Use LLM then HARD-OVERRIDE with rules
            cat = item.get("Category")
            is_debt = bool(item.get("IsDebtPayment", False))
            kind = item.get("DebtKind")
            name = item.get("DebtName")

            forced = _force_rules(r["description"], r["amount"])
            if forced:
                cat = forced["Category"]
                is_debt = forced["IsDebtPayment"]
                kind = forced["DebtKind"]
                name = forced["DebtName"]

            if cat not in ALLOWED_CATEGORIES:
                cat = "Other"
            if is_debt and kind not in ALLOWED_DEBT_KINDS:
                pk = _extract_parenthetical_kind(r["description"])
                kind = pk if pk in ALLOWED_DEBT_KINDS else None

            out.at[j, "Category"] = cat
            out.at[j, "IsDebtPayment"] = is_debt
            out.at[j, "DebtKind"] = kind
            out.at[j, "DebtName"] = name or re.sub(PAREN_KIND_RE, "", r["description"]).strip() or None

    return out

__all__ = [
    "categorize_with_llm",
    "ALLOWED_CATEGORIES",
    "ALLOWED_DEBT_KINDS",
]
