import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
# Import necessary variables from config
from config import embeddings, get_vector_store, set_vector_store

def get_loader_for_file(file_path):
    """Get appropriate loader based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return PyPDFLoader(file_path)
    elif ext == '.txt':
        # Try utf-8 first, fallback to latin-1 if needed
        try:
            return TextLoader(file_path, encoding='utf-8')
        except Exception as e:
            print(f"UTF-8 load failed for {file_path}: {e}, trying latin-1")
            return TextLoader(file_path, encoding='latin-1')
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def initialize_vector_store():
    """Initializes the vector store by loading documents from the 'documents' directory."""
    # Load all documents from the directory
    documents = []
    for filename in os.listdir('documents'):
        if filename.lower().endswith(('.txt', '.pdf')):
            file_path = os.path.join('documents', filename)
            try:
                loader = get_loader_for_file(file_path)
                loaded_docs = loader.load()
                if loaded_docs:
                    documents.extend(loaded_docs)
                    print(f"Successfully loaded {filename}")
                else:
                    print(f"Warning: No content loaded from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue

    if not documents:
        print("No documents found in the documents directory")
        set_vector_store(None)
        return None

    # Split documents into chunks with more strict splitting
    text_splitter = CharacterTextSplitter(
        chunk_size=500,  # Further reduced chunk size
        chunk_overlap=50,  # Reduced overlap
        length_function=len,
        separator="\n",  # Split on single newlines
        is_separator_regex=False
    )
    
    try:
        texts = text_splitter.split_documents(documents)
        print(f"Split documents into {len(texts)} chunks")
        
        # Verify chunk sizes
        for i, text in enumerate(texts):
            chunk_size = len(text.page_content)
            if chunk_size > 500:
                print(f"Warning: Chunk {i} size {chunk_size} exceeds limit")
                # Force split if too large
                text.page_content = text.page_content[:500]
        
        # Create vector store and assign it to the global variable
        store = FAISS.from_documents(texts, embeddings)
        set_vector_store(store)
        print(f"Successfully loaded {len(documents)} documents into vector store")
        return store
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        set_vector_store(None)
        return None

def add_texts_to_vector_store(texts):
    """Adds new texts to the existing vector store or creates a new one."""
    store = get_vector_store()
    if store is None:
        store = FAISS.from_documents(texts, embeddings)
        set_vector_store(store)
        print("Created new vector store.")
    else:
        store.add_documents(texts)
        print(f"Added {len(texts)} texts to existing vector store.")