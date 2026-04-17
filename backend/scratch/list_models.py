import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
if not api_key:
    # try config.yaml
    import yaml
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
            api_key = cfg.get('llm', {}).get('gemini', {}).get('api_key')

if not api_key:
    print("No API key found")
    exit(1)

genai.configure(api_key=api_key.strip())
print("Available models supporting generateContent:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
