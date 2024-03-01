import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from app.helpers import configReader

# Define config parameters
config = configReader()

def init_qdrant(collection_name: str) -> QdrantVectorStore:
    '''
    Initiate Qdrant environment
    '''
    client = qdrant_client.QdrantClient(
        url=config.get_url('qdrant'), 
        api_key=config.get_apikey('qdrant'),
    )
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

    return vector_store