import requests
from flask import jsonify

def process_chat(user_message, context):
    """
    Process chat request by sending it to the local language model API.
    
    Args:
        user_message (str): The user's message
        context (str): The context from vector store search
        
    Returns:
        tuple: (response_data, status_code) containing the chat response and status code
    """
    try:
        # Prepare the payload for the local API with context
        payload = {
            "model": "mathstral-7b-v0.1",  # Consider making this configurable
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the question: " + context},
                {"role": "user", "content": user_message}
            ]
        }

        # Make request to the local API (Consider making URL configurable)
        response = requests.post(
            'http://127.0.0.1:1234/v1/chat/completions',
            json=payload
        )
        response.raise_for_status()  # Raise an exception for bad status codes

        # Extract just the message content from the response
        response_data = response.json()
        assistant_message = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')

        # Return the message and the context used
        return jsonify({
            "message": assistant_message,
            "context_used": context
        }), 200

    except requests.exceptions.RequestException as req_err:
        print(f"API request error: {str(req_err)}")
        return jsonify({"error": f"Failed to connect to the local language model API: {str(req_err)}"}), 503
    except Exception as e:
        print(f"Error in chat processing: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500 