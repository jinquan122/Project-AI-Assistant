from app.controllers.qdrant.init_func import init_qdrant
from app.helpers import config_reader
from llama_index.core import set_global_service_context
from llama_index.core import get_response_synthesizer
from app.controllers.llamaindex.prompts import text_qa_template
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, TimeWeightedPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.llms.gemini import Gemini
# from traceloop.sdk import Traceloop
from app.helpers import configReader
import phoenix as px

# Define config parameters
config = configReader()
gemini_api_key = config.get_apikey('gemini')
# Traceloop.init(disable_batch=True, api_key=config.get('traceloop','api_key'))

# Define LLM and Embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Gemini(api_key=gemini_api_key, 
             model_name='models/gemini-pro', 
             temperature=0)
service_context = ServiceContext.from_defaults(embed_model = embed_model, llm=llm)
set_global_service_context(service_context)

def langchain_retriever(
    similarity_cutoff: int = 0.5,
    retrieve_top_k: int = 10
) -> RetrieverQueryEngine:
  '''
  Initialize the agent with three tools.
  1. Vector db retriever tool
  2. Google Search tool
  3. Plotting tool

  Args:
    model (str): The model to use for the agent.
    temperature (int): The temperature to use for the agent.
    seed (int): The seed to use for the agent.
    similarity_cutoff (int): The similarity cutoff to use for the agent.
    retrieve_top_k (int): The retrieve top k to use for the agent.

  Returns:
    OpenAIAgent: The agent with three tools.
  '''
  # Define vector db retriever tools
  storage_context = init_qdrant('news') 
  service_context = ServiceContext.from_defaults(embed_model = embed_model, llm=llm)
  index = VectorStoreIndex.from_vector_store(
    vector_store=storage_context.vector_store,
    service_context=service_context
    )
  retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=retrieve_top_k
  )
  node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=similarity_cutoff),
        TimeWeightedPostprocessor(time_decay=0.5, time_access_refresh=False, top_k=10)
    ]
  response_synthesizer = get_response_synthesizer(
    response_mode="compact",
    text_qa_template=text_qa_template,
    # refine_template=refine_template
  )
  vector_db_query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=node_postprocessors,
    response_synthesizer=response_synthesizer
  )
  
  return vector_db_query_engine