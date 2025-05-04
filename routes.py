from flask import Blueprint, request, jsonify
import requests
import os
from langchain.text_splitter import CharacterTextSplitter

# Import necessary functions and variables
from vector_store_utils import get_loader_for_file, add_texts_to_vector_store, initialize_vector_store
from config import get_vector_store
from chat_service import process_chat  # Import the new chat service

# Create a Blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/add-document', methods=['POST'])
def add_document():
    global global_vector_store # Ensure we are referencing the global store
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Check file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.txt', '.pdf']:
            return jsonify({"error": "Unsupported file type. Only .txt and .pdf files are allowed"}), 400

        # Ensure documents directory exists
        if not os.path.exists('documents'):
            os.makedirs('documents')

        # Save the file
        file_path = os.path.join('documents', file.filename)
        file.save(file_path)

        # Process and add to vector store
        try:
            loader = get_loader_for_file(file_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            # Add texts using the utility function which handles the global vector_store
            add_texts_to_vector_store(texts)

            return jsonify({"message": f"Document {file.filename} added successfully"})
        except Exception as e:
            # Clean up the file if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e # Re-raise the exception to be caught by the outer try-except

    except Exception as e:
        # Log the error for debugging
        print(f"Error in add_document: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@api_bp.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        store = get_vector_store()
        if store is None:
            print("Vector store not initialized, attempting initialization...")
            from vector_store_utils import initialize_vector_store
            store = initialize_vector_store()
            if store is None:
                return jsonify({"error": "Vector store is not available."}), 500

        relevant_docs = store.similarity_search(user_message, k=2)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        return process_chat(user_message, context)
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

def get_loader_for_file(file_path):
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