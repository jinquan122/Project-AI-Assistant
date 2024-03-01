from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SimilarityPostprocessor
import os
from app.helpers import configReader

# Read config parameters
config = configReader()

class PostPorcessor():
    def __init__(self):
        os.environ["COHERE_API_KEY"] = config.get_apikey('cohere')

    def reranker(self):
        '''
        Important for reanking the retreived nodes.
        '''
        return CohereRerank(top_n=10)
    
    def similarity_cutoff(self):
        '''
        Important for removing less relevant nodes.
        '''
        return SimilarityPostprocessor(similarity_cutoff=0.5)