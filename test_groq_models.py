# test_groq_models.py - Check what models are available
from groq import Groq
from config import GROQ_API_KEY

def test_available_models():
    """Test what models are available with your Groq API key."""
    client = Groq(api_key=GROQ_API_KEY)
    
    print("🔍 Testing Groq models...")
    
    # Common Groq model names to try
    models_to_test = [
        "llama3-70b-8192",
        "llama3-8b-8192", 
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ]
    
    working_models = []
    
    for model in models_to_test:
        try:
            print(f"Testing: {model}...")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            print(f"✅ {model} - WORKS!")
            working_models.append(model)
        except Exception as e:
            print(f"❌ {model} - Error: {str(e)[:100]}...")
    
    print(f"\n📊 SUMMARY:")
    if working_models:
        print(f"✅ Working models:")
        for model in working_models:
            print(f"   • {model}")
        print(f"\n🎯 Use this model in your pdf_extractor.py:")
        print(f"model=\"{working_models[0]}\",")
    else:
        print("❌ No models working - check your API key")

if __name__ == "__main__":
    test_available_models()