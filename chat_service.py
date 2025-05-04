import requests
import os
from flask import jsonify
from collections import deque
from typing import Deque, Dict

# Initialize a deque with maxlen=5 to store the last 5 conversations
conversation_history: Deque[Dict] = deque(maxlen=5)

def process_chat(user_message, context, context_info):
    """
    Process chat request by sending it to the local language model API.
    
    Args:
        user_message (str): The user's message
        context (str): The context from vector store search
        context_info (list): List of dictionaries containing source information
        
    Returns:
        tuple: (response_data, status_code) containing the chat response and status code
    """
    try:
        # Create messages list starting with system message
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the question: " + context},
        ]
        
        # Add conversation history to messages
        for hist in conversation_history:
            messages.append({"role": "user", "content": hist["user_message"]})
            messages.append({"role": "assistant", "content": hist["assistant_message"]})
            
        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Prepare the payload for the local API with context
        payload = {
            "model": "mathstral-7b-v0.1",
            "messages": messages
        }

        # Make request to the local API
        response = requests.post(
            'http://127.0.0.1:1234/v1/chat/completions',
            json=payload
        )
        response.raise_for_status()

        # Extract just the message content from the response
        response_data = response.json()
        assistant_message = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')

        # Store the conversation in history
        conversation_history.append({
            "user_message": user_message,
            "assistant_message": assistant_message
        })

        # Format source information
        sources = [{
            'file': os.path.basename(doc['source']),
            'page': doc['page'] if doc['page'] is not None else 'N/A'
        } for doc in context_info]

        # Return the message, context used, and source information
        return jsonify({
            "message": assistant_message,
            "context_used": context,
            "sources": sources,
            "history_length": len(conversation_history)
        }), 200

    except requests.exceptions.RequestException as req_err:
        print(f"API request error: {str(req_err)}")
        return jsonify({"error": f"Failed to connect to the local language model API: {str(req_err)}"}), 503
    except Exception as e:
        print(f"Error in chat processing: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500