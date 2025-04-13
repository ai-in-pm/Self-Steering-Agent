import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

def test_api_connection():
    """Test the connection to the OpenAI API."""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you hear me?"}
            ],
            max_tokens=50
        )
        print("API Connection Successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"API Connection Failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing OpenAI API connection...")
    test_api_connection()
