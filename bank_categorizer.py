import os
import pandas as pd
from openai import OpenAI
from config import DATA_DIRECTORY, OUTPUT_DIRECTORY, OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Ensure output directory exists
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

CATEGORIES = [
    "Rent/Mortgage",
    "Subscriptions",
    "Dining Out",
    "Transport",
    "Groceries",
    "Shopping",
    "Other",
    "Administrative"
]

def find_description_column(df):
    """Find the description column with different possible names."""
    possible_names = [
        'Description', 'description', 'DETAILS', 'Transaction Description',
        'Narration', 'Reference', 'Memo'
    ]
    for col in df.columns:
        if col in possible_names or any(name.lower() in col.lower() for name in possible_names):
            return col
    return None

def find_amount_column(df):
    """Find the amount column with different possible names."""
    possible_names = [
        'Amount', 'amount', 'AMOUNT', 'Amount (ZAR)', 'Debit', 'Credit',
        'Transaction Amount', 'Value'
    ]
    for col in df.columns:
        if col in possible_names or any(name.lower() in col.lower() for name in possible_names):
            return col
    return None

def classify_transaction(description, amount=None):
    """Classify a bank transaction description into predefined categories."""
    if pd.isna(description) or str(description).strip() == "":
        return "Other"
    
    desc_lower = str(description).lower()
    
    # Administrative/Banking fees
    if any(word in desc_lower for word in ['opening balance', 'account maintenance', 'atm fee', 'bank fee']):
        return "Administrative"
    
    # Income patterns
    if any(word in desc_lower for word in ['salary', 'wage', 'overtime', 'pay']):
        return "Other"  # Will be classified as income later
    
    # Rent/Mortgage patterns
    if any(word in desc_lower for word in ['rent', 'mortgage', 'room share', 'accommodation']):
        return "Rent/Mortgage"
    
    # Groceries patterns
    if any(word in desc_lower for word in ['shoprite', 'pick n pay', 'checkers', 'spar', 'woolworths', 'groceries', 'food store']):
        return "Groceries"
    
    # Dining Out patterns  
    if any(word in desc_lower for word in ['kfc', 'mcdonald', 'restaurant', 'takeaway', 'food', 'pizza', 'burger', 'nando']):
        return "Dining Out"
    
    # Transport patterns
    if any(word in desc_lower for word in ['taxi', 'uber', 'bolt', 'transport', 'fuel', 'petrol', 'bus fare']):
        return "Transport"
    
    # Subscriptions patterns
    if any(word in desc_lower for word in ['insurance', 'subscription', 'netflix', 'dstv', 'vodacom', 'mtn', 'airtime', 'data', 'electricity', 'prepaid electricity', 'water', 'municipal']):
        return "Subscriptions"
    
    # Shopping patterns
    if any(word in desc_lower for word in ['pep stores', 'clothing', 'edgars', 'truworths', 'shopping', 'clothes', 'fashion']):
        return "Shopping"
    
    # ATM withdrawals
    if 'atm withdrawal' in desc_lower:
        return "Other"
    
    # Return "Other" instead of calling AI
    return "Other"

def process_file(filepath):
    """Process a CSV file and add category classifications."""
    try:
        # Read CSV file
        df = pd.read_csv(filepath)
        
        print(f"📄 Processing file: {os.path.basename(filepath)}")
        print(f"📊 Columns found: {list(df.columns)}")
        
        # Find description and amount columns
        desc_col = find_description_column(df)
        amount_col = find_amount_column(df)
        
        if not desc_col:
            print(f"❌ No description column found in {filepath}")
            return False
        
        print(f"✅ Using '{desc_col}' as description column")
        if amount_col:
            print(f"✅ Using '{amount_col}' as amount column")
        else:
            print("ℹ️  No amount column found (proceeding without amounts)")
        
        # Initialize Category column
        df["Category"] = None
        
        # Classify each transaction
        for idx, row in df.iterrows():
            description = row.get(desc_col, '')
            amount = row.get(amount_col, None) if amount_col else None
            category = classify_transaction(description, amount)
            df.at[idx, "Category"] = category
        
        # Save categorized file
        output_filename = f"categorized_{os.path.basename(filepath)}"
        output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
        df.to_csv(output_path, index=False)
        
        print(f"✅ Saved categorized file: {output_filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return False

if __name__ == "__main__":
    # Example usage for testing
    sample_file = os.path.join(DATA_DIRECTORY, "sample1_min_wage_earner.csv")
    if os.path.exists(sample_file):
        process_file(sample_file)
    else:
        print(f"Sample file not found: {sample_file}")