from app.controllers.llamaindex.llms import LLM
from app.controllers.llamaindex.observability import LLM_observability
from app.controllers.llamaindex.postprocessors import PostPorcessor
from app.controllers.llamaindex.prompts import system_prompt
from app.controllers.llamaindex.response_synthesizers import ResponseSynthesizer
from app.controllers.llamaindex.retrievers import Retriever, CustomRetriever
from app.controllers.qdrant.init_func import init_qdrant
from llama_index.core import Settings, VectorStoreIndex
from app.helpers import configReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import get_response_synthesizer


# Initiate observability for the query pipeline
observability_module = LLM_observability()
observability_module.phoenix_observability()

# Define config parameters
config = configReader()
openai.api_key = config.get_apikey('openai')

# Initiate gemini as LLM in the query pipeline
# llm_module = LLM()
# llm = llm_module.gemini()
llm = OpenAI(model=config.get_model(), temperature=0)

# Global set LLM and Embedding model
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name=config.get_embed_model())

# Define retriever
vector_store = init_qdrant(collection_name=config.get_collection())
vector_index = VectorStoreIndex.from_vector_store(vector_store)
retriever_module = Retriever()
vector_retriever = retriever_module.vector_index_retriever(vector_index)
# bm25_retriever = BM25Retriever.from_defaults(index=vector_index)
# retriever = CustomRetriever(vector_retriever, bm25_retriever)

# Define node postprocessors
postprocessors_module = PostPorcessor()
reranker = postprocessors_module.reranker()
similarity_cutoff = postprocessors_module.similarity_cutoff()

# Define response synthesizer
response_synthesizer_module = ResponseSynthesizer()
summarizer = response_synthesizer_module.refine_summarize()

def init_agent() -> OpenAIAgent:
    '''
    Initialize React Agent to answer questions about Data Science Natural Language Processing.
    Returns:
        ReActAgent: React Agent to answer questions about Data Science Natural Language Processing.
    '''
    query_engine = RetrieverQueryEngine(
        retriever=vector_retriever,
        node_postprocessors=[reranker, similarity_cutoff],
        response_synthesizer=get_response_synthesizer(response_mode='compact'))
    
    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="Data_Science_Natural_Language_Processing_knowledge_base",
            description="Provides information about Data Science Natural Language Processing.",
        ),
    )

    return OpenAIAgent.from_tools(
    [query_engine_tool],
    llm=llm,
    verbose=True,
    system_prompt=system_prompt,
    max_function_calls=3,
    # context=system_prompt
    )







