# test_llm_extraction.py - Test bank statement extraction with different models
from groq import Groq
from config import GROQ_API_KEY
import sys

def test_extraction_with_models(sample_text):
    """Test bank statement extraction with different Groq models."""
    client = Groq(api_key=GROQ_API_KEY)
    
    # Models that worked in our previous test
    models_to_test = [
        "llama3-70b-8192",
        "llama3-8b-8192", 
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ]
    
    prompt = f"""Extract bank statement transaction data from the following text and convert it to CSV format.

Required CSV format (exact headers):
Date,Description,Amount (ZAR),Balance (ZAR)

Rules:
1. Date format: YYYY-MM-DD (convert from any date format you find)
2. For debit transactions: use negative amount (e.g., -50.00)
3. For credit transactions: use positive amount (e.g., 50.00)
4. Balance: always show the balance amount
5. Description: clean transaction description
6. Only return the CSV data, no explanations
7. Ignore privacy placeholders like [ACCOUNT_NUMBER], [PERSONAL_INFO_REMOVED]

Example output format:
Date,Description,Amount (ZAR),Balance (ZAR)
2025-08-01,Monthly facility fee,-50.00,-73190.00
2025-08-01,Interest Earned,0.34,83.34

Bank statement text:
{sample_text}

CSV output:
"""

    results = {}
    
    for model in models_to_test:
        print(f"\n{'='*60}")
        print(f"🤖 Testing model: {model}")
        print(f"{'='*60}")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a bank statement data extraction expert. Extract transaction data and return clean CSV format. Focus on dates, descriptions, amounts, and balances."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            results[model] = result
            
            print(f"✅ SUCCESS!")
            print(f"Response length: {len(result)} characters")
            print(f"CSV lines: {len(result.split(chr(10)))}")
            print(f"\nFirst 300 chars of result:")
            print("-" * 40)
            print(result[:300] + "..." if len(result) > 300 else result)
            print("-" * 40)
            
            # Save full result to file
            filename = f"result_{model.replace('-', '_').replace('.', '_')}.csv"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"💾 Full result saved to: {filename}")
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            results[model] = f"Error: {str(e)}"
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY OF ALL MODELS")
    print(f"{'='*60}")
    
    successful_models = []
    for model, result in results.items():
        if not result.startswith("Error:"):
            successful_models.append(model)
            lines = result.split('\n')
            csv_lines = [line for line in lines if ',' in line and not line.startswith('#')]
            print(f"✅ {model}: {len(csv_lines)} CSV lines extracted")
        else:
            print(f"❌ {model}: {result[:50]}...")
    
    if successful_models:
        print(f"\n🎯 RECOMMENDED MODEL: {successful_models[0]}")
        print(f"💡 Update your pdf_extractor.py with:")
        print(f'   model="{successful_models[0]}",')
    else:
        print(f"\n❌ No models worked successfully")
    
    return results

def get_sample_data():
    """Get sample data from the ai_preview.txt or use hardcoded sample."""
    try:
        with open('ai_preview.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract just a portion for testing (to avoid huge prompts)
        lines = content.split('\n')
        sample_lines = []
        in_transactions = False
        line_count = 0
        
        for line in lines:
            if 'Date' in line and 'Description' in line and 'Debit' in line:
                in_transactions = True
            
            if in_transactions and line_count < 20:  # Take first 20 transaction lines
                sample_lines.append(line)
                if line.strip():
                    line_count += 1
        
        sample = '\n'.join(sample_lines)
        if len(sample) > 100:
            print(f"📄 Using sample from ai_preview.txt ({len(sample)} chars)")
            return sample
        else:
            raise Exception("Sample too small")
            
    except:
        # Fallback to hardcoded sample
        sample = """Date
Description
Debit
Credit
Balance
[ACCOUNT_NUMBER]
Monthly facility fee
R 50.00
R 73,190.00-
[ACCOUNT_NUMBER]
Interest Charged at 21.50%
R 1,335.96
R 74,525.96-
[ACCOUNT_NUMBER]
ENGEN MALVERN CONV CEN Durban
R 200.00
R 74,840.96-
[ACCOUNT_NUMBER]
Interest Earned at 5.00%
R 0.34
R 82.67"""
        
        print(f"📄 Using hardcoded sample data")
        return sample

if __name__ == "__main__":
    print("🧪 TESTING LLM EXTRACTION WITH DIFFERENT MODELS")
    print("=" * 60)
    
    # Get sample data
    sample_text = get_sample_data()
    print(f"Sample text preview (first 200 chars):")
    print("-" * 40)
    print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)
    print("-" * 40)
    
    # Test all models
    results = test_extraction_with_models(sample_text)
    
    print(f"\n🎉 Testing completed!")
    print(f"📁 Check the generated CSV files for full results")