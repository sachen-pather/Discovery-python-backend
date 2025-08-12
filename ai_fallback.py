"""
ai_fallback.py - Simplified AI fallback for categorization
Compatible with your existing infrastructure
"""
import json
import os
import re
import time
from typing import Dict, List, Tuple

# Try to import OpenAI SDK
try:
    from openai import OpenAI
    from config import OPENAI_API_KEY
    AI_AVAILABLE = bool(OPENAI_API_KEY)
    if AI_AVAILABLE:
        client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    AI_AVAILABLE = False
    client = None

# Enhanced rule-based classification
CATEGORY_RULES = [
    # Administrative
    (re.compile(r"^opening balance$", re.I), "Administrative"),
    (re.compile(r"account maintenance|bank fee|atm fee", re.I), "Administrative"),
    
    # Income
    (re.compile(r"salary|wage|payroll|overtime|bonus", re.I), "Income"),
    (re.compile(r"sassa|grant|pension", re.I), "Income"),
    
    # Housing
    (re.compile(r"rent|mortgage|bond|room share|accommodation", re.I), "Rent/Mortgage"),
    
    # Groceries
    (re.compile(r"shoprite|pick\s*n\s*pay|checkers|spar|woolworths", re.I), "Groceries"),
    (re.compile(r"grocer(ies)?", re.I), "Groceries"),
    
    # Dining Out
    (re.compile(r"kfc|mcdonald|nando|restaurant|takeaway|pizza", re.I), "Dining Out"),
    
    # Transport
    (re.compile(r"uber|taxi|petrol|fuel|transport|bus fare", re.I), "Transport"),
    
    # Subscriptions
    (re.compile(r"insurance|vodacom|mtn|cell c|telkom|airtime|data", re.I), "Subscriptions"),
    (re.compile(r"electricity|water|municipal|netflix|dstv", re.I), "Subscriptions"),
    
    # Shopping
    (re.compile(r"pep stores|ackermans|edgars|truworths|clothing|clothes", re.I), "Shopping"),
    
    # Other patterns
    (re.compile(r"withdrawal|atm", re.I), "Other"),
    (re.compile(r"transfer|money transfer", re.I), "Other"),
]

def rules_classify(description: str) -> Tuple[str, float, str]:
    """
    Classify a transaction description using rules.
    Returns (category, confidence, reason)
    """
    if not description or pd.isna(description):
        return "Other", 0.3, "empty_description"
    
    desc_clean = str(description).strip().lower()
    
    for pattern, category in CATEGORY_RULES:
        if pattern.search(desc_clean):
            return category, 0.9, f"rule:{pattern.pattern[:20]}"
    
    return "Other", 0.4, "no_rule_match"

def classify_with_ai_batch(descriptions: List[str]) -> Dict[str, str]:
    """
    Classify descriptions using AI in batch.
    Falls back to rules if AI is not available.
    """
    if not AI_AVAILABLE or not client:
        # Fallback to rules only
        return {desc: rules_classify(desc)[0] for desc in descriptions}
    
    # Filter out descriptions that are already well-classified by rules
    high_confidence_results = {}
    uncertain_descriptions = []
    
    for desc in descriptions:
        category, confidence, reason = rules_classify(desc)
        if confidence >= 0.8:
            high_confidence_results[desc] = category
        else:
            uncertain_descriptions.append(desc)
    
    # If no uncertain descriptions, return rule-based results
    if not uncertain_descriptions:
        return high_confidence_results
    
    # Process uncertain descriptions with AI
    ai_results = {}
    try:
        # Build prompt
        categories = ["Income", "Groceries", "Transport", "Rent/Mortgage", 
                     "Subscriptions", "Shopping", "Administrative", "Other", "Dining Out"]
        
        descriptions_text = "\n".join(f"- {desc}" for desc in uncertain_descriptions)
        
        prompt = f"""Categorize these bank transaction descriptions into exactly one of: {', '.join(categories)}.
Return JSON with description as key and category as value.

Descriptions:
{descriptions_text}

JSON:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a bank transaction categorizer. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse AI response
        try:
            ai_data = json.loads(response_text)
            for desc in uncertain_descriptions:
                ai_results[desc] = ai_data.get(desc, "Other")
        except json.JSONDecodeError:
            # If JSON parsing fails, use rules
            ai_results = {desc: rules_classify(desc)[0] for desc in uncertain_descriptions}
            
    except Exception as e:
        print(f"⚠️ AI classification failed: {e}")
        # Fallback to rules for uncertain descriptions
        ai_results = {desc: rules_classify(desc)[0] for desc in uncertain_descriptions}
    
    # Combine results
    final_results = {**high_confidence_results, **ai_results}
    return final_results

def classify_with_rules_then_ai(descriptions: List[str]) -> Dict[str, str]:
    """
    Main function for classification with rules-first approach.
    Compatible with your friend's enhanced categorizer.
    """
    if isinstance(descriptions, str):
        descriptions = [descriptions]
    
    # Remove duplicates while preserving order
    unique_descriptions = list(dict.fromkeys(descriptions))
    
    # Batch classify
    results = classify_with_ai_batch(unique_descriptions)
    
    # Ensure all descriptions have results
    for desc in unique_descriptions:
        if desc not in results:
            results[desc] = "Other"
    
    return results

# For compatibility with pandas
try:
    import pandas as pd
except ImportError:
    # Create a minimal mock for pd.isna if pandas not available
    class MockPandas:
        @staticmethod
        def isna(value):
            return value is None or (isinstance(value, str) and value.strip() == "")
    pd = MockPandas()