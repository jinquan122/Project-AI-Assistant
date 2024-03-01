from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser, SemanticSplitterNodeParser
from app.helpers import configReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from nltk.tokenize import sent_tokenize
from typing import Callable, List

# Define config parameters
config = configReader()

def _sentence_tokenizer_splitter() -> Callable[[str], List[str]]:
    def split(text:str) -> List[str]:
        sentences = sent_tokenize(text)
        return sentences
    return split

class NodeParser:
    '''
    This class is used to parse articles into sentences or other formats which can be better retrieved by RAG.
    '''
    def __init__(self):
        None

    def sentence_spliter(self):
        '''
        This function is used to split the text into sentences.
        '''
        return SentenceSplitter(
            chunk_size=512, 
            chunk_overlap=25, 
            chunking_tokenizer_fn=_sentence_tokenizer_splitter())
    
    def markdown_parser(self):
        '''
        This function is used to parse the text into markdown format.
        '''
        return MarkdownNodeParser()
    
    def semantic_spliter(self):
        '''
        This function is to group all semantic sentences into chunks.
        '''
        return SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=95, 
            embed_model=HuggingFaceEmbedding(model_name=config.get_embed_model()))