# backend/config.py
# Configuration file for Bank Transaction Categorizer
# IMPORTANT: Keep this file private and do not share it!
# Add this file to .gitignore if using version control
import os  # This import was missing

# Your OpenAI API Key

OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "categorized_output")# Your OpenAI API Key
OPENAI_API_KEY = "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"

# Relative directories based on app.py location
DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")
