import os, traceback
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")


from llm_service import LLMService

svc = LLMService()
if svc.llm:
    try:
        result = svc.generate_question('Photosynthesis is the process by which plants convert sunlight into food using CO2 and water.')
        print('SUCCESS:', result)
    except Exception as e:
        traceback.print_exc()
else:
    print('LLM not initialized - check API key')
