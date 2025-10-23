from groq import Groq
import os
import dotenv
dotenv.load_dotenv()


def get_chatbot_response(client, model_name, messages, temperature=0):
    input_messages = []
    for message in messages:
        input_messages.append({"role": message["role"], "content": message["content"]})
    response = client.chat.completions.create(
        model=model_name,
        messages=input_messages,
        temperature=temperature,
        top_p=0.8,
        max_tokens=2000
    ).choices[0].message.content

    return response

# Get Embeddings
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def get_embedding(embedding_client, embedding_model_name, text_input):
    """
    Get embeddings locally (384 dimensions with batch_size=32)
    
    Args:
        embedding_client: HuggingFaceBgeEmbeddings instance
        model_name: Model name (BAAI/bge-small-en-v1.5)
        text_input: Text or list of texts to embed
    
    Returns:
        List of 384-dimensional embeddings
    """
    if isinstance(text_input, str):
        text_input = [text_input]
    
    embeddings = []
    for text in text_input:
        embedding = embedding_client.embed_query(text)
        embeddings.append(embedding)
    
    return embeddings


def double_check_json_output(client, model_name, json_string):
    prompt = f""" You will check this json string and correct any mistakes that will make it invalid. Then you will return the corrected json string. Nothing else. 
    If the Json is correct just return it.

    If there us any text before order after the json string, remove it. 
    Do NOT return a single letter outside of the json string.
    The first thing you should write should be open curly brace of the json and the last letter you write should be the closing curly brace.

    You should check the json string for the following text between tripe backticks:
    ```
    {json_string}
    ```
    """

    messages = [{"role":"user","content":prompt}]
    response = get_chatbot_response(client, model_name, messages)
    response = response.replace("`","")
    
    return response
