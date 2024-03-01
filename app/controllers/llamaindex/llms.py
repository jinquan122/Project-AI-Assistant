from llama_index.llms.gemini import Gemini
from app.helpers import configReader

# Read config parameters
config = configReader()

class LLM:
    '''
    Define LLM for RAG. Currently only Gemini API is available.
    '''
    def __init__(self):
        self.gemini_api_key = config.get_apikey('gemini')

    def gemini(
            self,
            model_name: str = 'models/gemini-pro',
            temperature: float = 0
    ):
        llm = Gemini(api_key=self.gemini_api_key, 
                    model_name=model_name, 
                    temperature=temperature)
        return llm