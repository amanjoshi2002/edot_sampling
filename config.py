from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Global vector store variable
vector_store = None

def get_vector_store():
    global vector_store
    return vector_store

def set_vector_store(store):
    global vector_store
    vector_store = store