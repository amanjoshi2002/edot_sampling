from flask import Flask
# Import necessary functions and blueprints
from vector_store_utils import initialize_vector_store
from routes import api_bp # Import the blueprint

# Create Flask app instance
app = Flask(__name__)

# Register the blueprint
app.register_blueprint(api_bp)

if __name__ == '__main__':
    # Initialize the vector store when starting the application
    print("Initializing vector store...")
    initialize_vector_store()
    print("Starting Flask server...")
    # Run the Flask app
    app.run(port=5000, debug=True) # Added debug=True for development