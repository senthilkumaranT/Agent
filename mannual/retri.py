from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
import uuid
import textwrap

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="", 
    api_key="",
)

# Initialize OpenAI client
openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="")
# Collection name
COLLECTION_NAME = "document_chunks_2"

def create_collection_if_not_exists():
    """Create a collection if it doesn't exist"""
    collections = qdrant_client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if COLLECTION_NAME not in collection_names:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")


def get_embedding(text):
    """Get embedding for a text using SentenceTransformer"""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # If text is a single string, wrap it in a list
    if isinstance(text, str):
        embedding_vector = model.encode([text])[0]
    else:
        embedding_vector = model.encode(text)[0]
    
    return embedding_vector


def upload_document(document, chunk_size=1000):
    """Upload a document by splitting it into chunks and storing in Qdrant"""
    # Create collection if it doesn't exist
    create_collection_if_not_exists()
    
    # Split document into chunks
    chunks = textwrap.wrap(document, chunk_size, break_long_words=False, replace_whitespace=False)
    
    points = []
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        embedding = get_embedding(chunk)
        
        points.append(PointStruct(
            id=chunk_id,
            vector=embedding,
            payload={"text": chunk, "chunk_index": i, "document_id": str(uuid.uuid4())}
        ))
    
    # Upload chunks to Qdrant
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    print(f"Uploaded {len(chunks)} chunks to Qdrant")
    return len(chunks)


def retrieve_chunks(query, limit=5):
    """Retrieve relevant document chunks based on a query"""
    query_embedding = get_embedding(query)
    
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=limit
    )
    
    results = []
    for result in search_result:
        results.append({
            "text": result.payload["text"],
            "score": result.score,
            "chunk_index": result.payload["chunk_index"],
            "document_id": result.payload["document_id"]
        })
    
    return results

# Example usage
if __name__ == "__main__":
    print("Current collections:", qdrant_client.get_collections().collections)
    # create_collection_if_not_exists()

   
    sample_doc = "This is a sample document that contains information about vector databases. Qdrant is a vector database that allows for efficient similarity search."
    upload_document(sample_doc)
    # results = retrieve_chunks("Tell me about databases")
    # for i, result in enumerate(results):
    #     print(f"Result {i+1}: {result['text']} (Score: {result['score']})")