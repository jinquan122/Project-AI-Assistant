from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from app.controllers.llamaindex.node_parser import NodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.controllers.qdrant.init_func import init_qdrant
from app.helpers import configReader
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from app.controllers.llamaindex.transformations import TextCleaner
from llama_index.core.extractors import SummaryExtractor
from llama_index.extractors.entity import EntityExtractor
from app.controllers.llamaindex.llms import LLM

config = configReader()
# Initiate gemini as LLM in the query pipeline
llm_module = LLM()
llm = llm_module.gemini()

def pdf_ingest(
        embedding_model:HuggingFaceEmbedding = HuggingFaceEmbedding(config.get_embed_model())
        ) -> None:
    '''
    Ingest data(doc) into Qdrant vector store.

    Args:
    - doc: data in Document format (LlamaIndex Document pack).
    - embedding_model: embedding model to be used for indexing.

    Returns:
    None.

    Ingestion Pipeline:
    1. Parse the data into hierarchical format.
    2. Embed the data using embedding model.
    3. Store the data into Qdrant vector store.
    4. Create a vector store index from the vector store.
    5. Update the index with the new data.
    '''
    parser = LlamaParse(
        api_key=config.get_apikey('llamaparse'), 
        result_type="markdown"  # "markdown" and "text" are available
    )

    file_extractor = {".pdf": parser}
    reader = SimpleDirectoryReader("./data", file_extractor=file_extractor)
    docs = reader.load_data()

    # Initialize vector store and load node parser
    vector_store = init_qdrant(collection_name=config.get_collection())
    node_parser = NodeParser()

    # Build Ingestion Pipeline
    pipeline = IngestionPipeline(
        transformations=[
            TextCleaner(),
            node_parser.sentence_spliter(),
            # SummaryExtractor(summaries=["prev", "self", "next"], llm=llm),
            embedding_model,
        ],
        vector_store=vector_store,
    )

    # Ingest data(doc) into Qdrant vector db
    pipeline.run(documents=docs)

