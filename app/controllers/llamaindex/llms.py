from llama_index.llms.gemini import Gemini
from app.helpers import configReader
from google.generativeai.types import HarmCategory, HarmBlockThreshold, HarmProbability

# Read config parameters
config = configReader()

class LLM:
    '''
    Define LLM for RAG. Currently only Gemini API is available.
    '''
    def __init__(self):
        self.gemini_api_key = config.get_apikey('gemini')
        self.safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

    def gemini(
            self,
            model_name: str = 'models/gemini-pro',
            temperature: float = 0
    ):
        llm = Gemini(api_key=self.gemini_api_key, 
                    model_name=model_name, 
                    temperature=temperature,
                    safety_settings=self.safety_settings,)
        return llm