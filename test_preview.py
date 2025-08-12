# test_preview.py - Preview what would be sent to AI (same logic as main extractor)
import fitz  # PyMuPDF
import re
import sys
import os

def extract_text_from_pdf(pdf_path):
    """Extract all text from PDF - SAME AS MAIN EXTRACTOR."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return None

def filter_sensitive_data(text):
    """
    EXACT SAME filtering logic that will be added to main extractor.
    Remove sensitive personal information before sending to AI.
    """
    print("🔒 Filtering sensitive data for privacy protection...")
    
    filtered_text = text
    
    # 1. Remove account numbers (various formats)
    # Pattern: 12345678901, 1234-5678-9012, 1234 5678 9012
    filtered_text = re.sub(r'\b\d{4,}[-\s]?\d{4,}[-\s]?\d{4,}\b', '[ACCOUNT_NUMBER]', filtered_text)
    filtered_text = re.sub(r'\b\d{8,}\b', '[ACCOUNT_NUMBER]', filtered_text)
    
    # 2. Remove branch codes and swift codes
    filtered_text = re.sub(r'\b[A-Z]{4}[A-Z0-9]{2}([A-Z0-9]{3})?\b', '[BANK_CODE]', filtered_text)
    filtered_text = re.sub(r'\b\d{6}\b', '[BRANCH_CODE]', filtered_text)
    
    # 3. Remove South African ID numbers (13 digits)
    filtered_text = re.sub(r'\b\d{13}\b', '[ID_NUMBER]', filtered_text)
    
    # 4. Remove email addresses
    filtered_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', filtered_text)
    
    # 5. Remove phone numbers (various formats)
    filtered_text = re.sub(r'\b(?:\+27|0)(?:\d{2}\s?\d{3}\s?\d{4}|\d{9})\b', '[PHONE]', filtered_text)
    
    # 6. Process line by line for more specific filtering
    lines = filtered_text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        original_line = line.strip()
        
        # Skip or replace lines containing sensitive keywords
        sensitive_keywords = [
            'account number', 'account no', 'acc no', 'account:', 'a/c:',
            'account holder', 'name:', 'customer name', 'client name',
            'address', 'residential address', 'postal address',
            'id number', 'identity number', 'passport',
            'branch code', 'swift code', 'sort code',
            'phone number', 'mobile', 'cell', 'telephone',
            'email address', 'e-mail'
        ]
        
        # Check if line contains sensitive information
        is_sensitive = any(keyword in line_lower for keyword in sensitive_keywords)
        
        if is_sensitive:
            # Replace with placeholder
            filtered_lines.append('[PERSONAL_INFO_REMOVED]')
            continue
        
        # Skip lines that are just account numbers or ID numbers
        digits_only = re.sub(r'[^0-9]', '', original_line)
        if len(digits_only) >= 8 and digits_only.isdigit() and len(original_line.strip()) < 20:
            filtered_lines.append('[ACCOUNT_NUMBER]')
            continue
        
        # Keep the line if it's not sensitive
        filtered_lines.append(original_line)
    
    filtered_result = '\n'.join(filtered_lines)
    
    # 7. Remove any remaining long number sequences that might be sensitive
    filtered_result = re.sub(r'\b\d{7,}\b', '[NUMBER_REDACTED]', filtered_result)
    
    print(f"✅ Sensitive data filtering completed")
    return filtered_result

def generate_ai_prompt(filtered_text):
    """Generate the exact prompt that would be sent to OpenAI."""
    return f"""Extract bank statement transaction data from the following text and convert it to CSV format.

PRIVACY NOTE: Personal information has been filtered out for privacy protection.
Focus only on extracting transaction data (dates, descriptions, amounts, balances).

Required CSV format (exact headers):
Date,Description,Amount (ZAR),Balance (ZAR)

Rules:
1. Date format: YYYY-MM-DD
2. Amount: positive for income/credits, negative for expenses/debits, empty string for opening balance
3. Balance: always show the balance amount
4. Description: clean transaction description (ignore privacy placeholders)
5. Include opening balance row if present
6. Only return the CSV data, no explanations
7. Ignore lines with [ACCOUNT_NUMBER], [PERSONAL_INFO_REMOVED], etc.

Example output format:
Date,Description,Amount (ZAR),Balance (ZAR)
2025-07-01,Opening Balance,,150.0
2025-07-01,Account Maintenance Fee,-5.0,145.0
2025-07-01,Salary – Acme Co,5600.0,5745.0

Bank statement text (privacy filtered):
{filtered_text}

CSV output:
"""

def preview_everything(pdf_path):
    """
    Show comprehensive preview of what happens during AI processing.
    This simulates the exact same process as the main extractor.
    """
    print("🔍 COMPREHENSIVE AI PROCESSING PREVIEW")
    print("=" * 70)
    print("This shows you exactly what would happen when using AI extraction")
    print("Same logic that will be added to your main pdf_extractor.py")
    print("=" * 70)
    
    # Step 1: Extract text from PDF
    print(f"\n📄 STEP 1: Extracting text from PDF...")
    print(f"File: {pdf_path}")
    
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        return False
    
    print("✅ Text extraction successful")
    print(f"📊 Original text length: {len(pdf_text):,} characters")
    print(f"📊 Original text lines: {len(pdf_text.splitlines())}")
    
    # Step 2: Show original text sample
    print(f"\n📝 STEP 2: Original text sample (first 300 chars):")
    print("-" * 50)
    print(pdf_text[:300] + "..." if len(pdf_text) > 300 else pdf_text)
    print("-" * 50)
    
    # Step 3: Apply privacy filtering
    print(f"\n🔒 STEP 3: Applying privacy filtering...")
    filtered_text = filter_sensitive_data(pdf_text)
    
    print(f"📊 Filtered text length: {len(filtered_text):,} characters")
    print(f"📊 Filtered text lines: {len(filtered_text.splitlines())}")
    
    # Step 4: Show filtering statistics
    print(f"\n📈 STEP 4: Privacy filtering results:")
    print("-" * 40)
    stats = {
        'Account numbers': filtered_text.count('[ACCOUNT_NUMBER]'),
        'Personal info lines': filtered_text.count('[PERSONAL_INFO_REMOVED]'),
        'ID numbers': filtered_text.count('[ID_NUMBER]'),
        'Email addresses': filtered_text.count('[EMAIL]'),
        'Phone numbers': filtered_text.count('[PHONE]'),
        'Bank codes': filtered_text.count('[BANK_CODE]'),
        'Branch codes': filtered_text.count('[BRANCH_CODE]'),
        'Other numbers': filtered_text.count('[NUMBER_REDACTED]')
    }
    
    total_redactions = sum(stats.values())
    for item, count in stats.items():
        if count > 0:
            print(f"  🔒 {item}: {count}")
    
    print(f"  📊 Total redactions: {total_redactions}")
    if total_redactions == 0:
        print("  ⚠️  No sensitive data detected - review filtering rules")
    
    # Step 5: Show filtered text (what goes to AI)
    print(f"\n🤖 STEP 5: Text that would be sent to OpenAI:")
    print("=" * 60)
    print(filtered_text)
    print("=" * 60)
    
    # Step 6: Generate and show complete AI prompt
    print(f"\n📤 STEP 6: Complete AI prompt preview:")
    prompt = generate_ai_prompt(filtered_text)
    print("-" * 60)
    print(prompt)
    print("-" * 60)
    
    # Step 7: Save everything to files for review
    print(f"\n💾 STEP 7: Saving files for review...")
    base_name = pdf_path.replace('.pdf', '')
    
    files_created = []
    
    try:
        # Save original text
        original_file = f"{base_name}_1_original.txt"
        with open(original_file, 'w', encoding='utf-8') as f:
            f.write("ORIGINAL PDF TEXT\n")
            f.write("=" * 50 + "\n\n")
            f.write(pdf_text)
        files_created.append(original_file)
        
        # Save filtered text
        filtered_file = f"{base_name}_2_filtered.txt"
        with open(filtered_file, 'w', encoding='utf-8') as f:
            f.write("FILTERED TEXT (SENT TO AI)\n")
            f.write("=" * 50 + "\n\n")
            f.write(filtered_text)
        files_created.append(filtered_file)
        
        # Save complete AI prompt
        prompt_file = f"{base_name}_3_ai_prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write("COMPLETE AI PROMPT\n")
            f.write("=" * 50 + "\n\n")
            f.write(prompt)
        files_created.append(prompt_file)
        
        # Save comparison and analysis
        analysis_file = f"{base_name}_4_analysis.txt"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("PRIVACY FILTERING ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Original text length: {len(pdf_text):,} characters\n")
            f.write(f"Filtered text length: {len(filtered_text):,} characters\n")
            f.write(f"Data removed: {len(pdf_text) - len(filtered_text):,} characters\n\n")
            f.write("Redactions made:\n")
            for item, count in stats.items():
                if count > 0:
                    f.write(f"  - {item}: {count}\n")
            f.write(f"\nTotal redactions: {total_redactions}\n")
        files_created.append(analysis_file)
        
        print("✅ Files saved successfully:")
        for i, file in enumerate(files_created, 1):
            print(f"   {i}. {file}")
            
    except Exception as e:
        print(f"⚠️ Error saving files: {e}")
    
    # Step 8: Summary and recommendations
    print(f"\n📋 STEP 8: Summary and recommendations:")
    print("-" * 50)
    if total_redactions > 0:
        print("✅ Privacy filtering is working correctly")
        print(f"🔒 {total_redactions} sensitive items were protected")
        print("👍 Safe to use AI extraction with this filtering")
    else:
        print("⚠️  No sensitive data was filtered")
        print("🔍 Review your PDF and filtering rules")
        print("📝 You may need to adjust the filtering patterns")
    
    print(f"\n🎯 Next steps:")
    print("1. Review the generated files above")
    print("2. Check if filtering is appropriate for your needs")
    print("3. If satisfied, add this filtering to your main pdf_extractor.py")
    print("4. Test with --no-ai flag first, then enable AI extraction")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            print(f"📂 Current directory: {os.getcwd()}")
            print(f"📋 PDF files in current directory:")
            try:
                pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
                if pdf_files:
                    for file in pdf_files:
                        print(f"   📄 {file}")
                else:
                    print("   (No PDF files found)")
            except:
                print("   (Could not list files)")
            sys.exit(1)
        
        preview_everything(pdf_path)
        
    else:
        print("TEST PREVIEW - See what would be sent to AI")
        print("=" * 50)
        print("Usage: python test_preview.py <pdf_file_path>")
        print("")
        print("This script shows you:")
        print("  • Original PDF text")
        print("  • What gets filtered out for privacy")
        print("  • Exact text that would go to OpenAI")
        print("  • Complete AI prompt")
        print("  • Analysis files for review")
        print("")
        print("Example:")
        print('  python test_preview.py "CertifiedStatements.pdf"')
        print("")
        print("📂 Current directory:", os.getcwd())
        print("📋 PDF files available:")
        try:
            pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
            if pdf_files:
                for file in pdf_files:
                    print(f"   📄 {file}")
            else:
                print("   (No PDF files found)")
        except:
            print("   (Could not list files)")