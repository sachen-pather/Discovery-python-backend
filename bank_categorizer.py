# bank_categorizer.py
import os
import pandas as pd
from typing import Optional
from config import OUTPUT_DIRECTORY
# If you put the LLM helper in a separate file (recommended):
# from llm_categorizer import categorize_with_llm
#
# If you temporarily pasted the LLM code at the bottom of app.py,
# you can switch to rule-based only for now by setting USE_LLM = False.

USE_LLM = True
try:
    from llm_categorizer import categorize_with_llm
except Exception:
    USE_LLM = False

CATEGORIES = [
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

def _find_description_column(df: pd.DataFrame) -> Optional[str]:
    possible = [
        "Description", "description", "DETAILS", "Transaction Description",
        "Narration", "Reference", "Memo"
    ]
    for col in df.columns:
        if col in possible or any(p.lower() in str(col).lower() for p in possible):
            return col
    return None

def _find_amount_column(df: pd.DataFrame) -> Optional[str]:
    possible = [
        "Amount", "amount", "AMOUNT", "Amount (ZAR)", "Debit", "Credit",
        "Transaction Amount", "Value"
    ]
    for col in df.columns:
        if col in possible or any(p.lower() in str(col).lower() for p in possible):
            return col
    return None

def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() in {"date", "transaction date", "posting date"}:
            return c
    return None

# -------- Simple rule-based fallback (used if USE_LLM is False) --------
def _rule_classify(df: pd.DataFrame, desc_col: str, amount_col: Optional[str]) -> pd.DataFrame:
    import numpy as np
    out = df.copy()
    out["Category"] = None
    out["IsDebtPayment"] = False
    out["DebtKind"] = None
    out["DebtName"] = None

    def classify(desc, amt):
        text = str(desc or "").lower()
        if any(w in text for w in ["salary", "wage", "overtime", "pay", "payout", "deposit"]) or (
            amount_col and pd.notna(amt) and float(amt) > 0
        ):
            return "Income"
        if any(w in text for w in ["opening balance", "account maintenance", "atm fee", "bank fee"]):
            return "Administrative"
        debt_trigs = ["monthly payment", "repayment", "installment", "instalment", "debit order", "debit-order", "debitorder"]
        if any(t in text for t in debt_trigs) or "(" in text and ")" in text:
            # crude debt flag, finer details not needed for fallback
            return "Debt Payments"
        if any(w in text for w in ["rent", "room share", "accommodation"]) or "mortgage" in text:
            return "Rent/Mortgage"
        if any(w in text for w in ["shoprite", "pick n pay", "checkers", "spar", "woolworths", "groceries", "food store"]):
            return "Groceries"
        if any(w in text for w in ["kfc", "mcdonald", "restaurant", "takeaway", "pizza", "burger", "nando"]):
            return "Dining Out"
        if any(w in text for w in ["taxi", "uber", "bolt", "transport", "fuel", "petrol", "bus fare"]):
            return "Transport"
        if any(w in text for w in ["insurance", "subscription", "netflix", "dstv", "vodacom", "mtn", "airtime", "data", "electricity", "prepaid electricity", "water", "municipal"]):
            return "Subscriptions"
        if any(w in text for w in ["pep stores", "clothing", "edgars", "truworths", "shopping", "clothes", "fashion"]):
            return "Shopping"
        return "Other"

    if amount_col and amount_col in out.columns:
        out[amount_col] = pd.to_numeric(out[amount_col], errors="coerce")

    for i, row in out.iterrows():
        cat = classify(row[desc_col], row[amount_col] if amount_col else None)
        out.at[i, "Category"] = cat
        if cat == "Debt Payments":
            out.at[i, "IsDebtPayment"] = True
            out.at[i, "DebtName"] = row[desc_col]
    return out

# -------- PUBLIC API used by app.py --------
def process_file(filepath: str) -> bool:
    """
    Read CSV at `filepath`, categorize transactions, write categorized CSV to
    OUTPUT_DIRECTORY/categorized_<basename>, and return True/False.
    """
    try:
        df = pd.read_csv(filepath)
        desc_col = _find_description_column(df)
        if not desc_col:
            print(f"❌ No description column in {filepath}")
            return False
        amount_col = _find_amount_column(df)
        date_col = _find_date_column(df)

        if USE_LLM:
            print("🤖 Using LLM categorizer for transaction classification")
            from llm_categorizer import categorize_with_llm  # local import to avoid hard dep if missing
            categorized = categorize_with_llm(
                df,
                desc_col=desc_col,
                amount_col=amount_col,
                date_col=date_col,
                provider="openai",  # "openai" | "groq" | "auto"
                batch_size=25,
            )
        else:
            print("🧠 Using rule-based fallback categorizer")
            categorized = _rule_classify(df, desc_col, amount_col)

        # Persist categorized CSV for the rest of the pipeline
        out_name = f"categorized_{os.path.basename(filepath)}"
        out_path = os.path.join(OUTPUT_DIRECTORY, out_name)
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        categorized.to_csv(out_path, index=False)
        print(f"✅ Saved categorized file: {out_name}")

        # Optional: export a debts-only view the optimizer can use
        debts = categorized[categorized.get("IsDebtPayment") == True]
        if not debts.empty:
            debts_out = os.path.join(OUTPUT_DIRECTORY, "debts_from_statement.csv")
            # Minimal useful export
            cols = []
            for c in ["Date", desc_col, amount_col, "DebtKind", "DebtName", "Category"]:
                if c and c in categorized.columns and c not in cols:
                    cols.append(c)
            debts[cols].to_csv(debts_out, index=False)
            print(f"🧮 Extracted debts file: {os.path.basename(debts_out)}")

        return True
    except Exception as e:
        print(f"❌ Error in process_file: {e}")
        return False
