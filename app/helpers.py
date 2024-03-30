import configparser

class configReader():
    '''
    Read the config file and return the config object.
    Returns: 
        config object.
    '''
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

    def get_apikey(self, target:str):
        if target == 'gemini':
            apikey = self.config.get('gemini','api_key')
        elif target == 'qdrant':
            apikey = self.config.get('qdrant', 'api_key')
        elif target == 'cohere':
            apikey = self.config.get('cohere', 'api_key')
        elif target == 'openai':
            apikey = self.config.get('openai', 'api_key')
        elif target == 'llamaparse':
            apikey = self.config.get('llamaparse', 'api_key')
        elif target == 'groq':
            apikey = self.config.get('groq', 'api_key')
        return apikey
    
    def get_url(self, target:str):
        if target == 'qdrant':
            url = self.config.get('qdrant', 'url')
        return url
    
    def get_model(self):
        model = self.config.get('parameter', 'openai_llm_model')
        return model
    
    def get_embed_model(self):
        model = self.config.get('parameter', 'embed_model')
        return model
    
    def get_collection(self):
        collection = self.config.get('parameter', 'qdrant_collection')
        return collection
    

    