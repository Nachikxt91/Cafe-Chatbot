from dotenv import load_dotenv
import os
from .utils import get_chatbot_response
from groq import Groq
from copy import deepcopy
from pinecone import Pinecone   
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import

load_dotenv()

class DetailsAgent():
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv('GROQ_API_KEY')
        )
        self.model_name = os.getenv("MODEL_NAME")
        
        self.embedding_client = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )
        
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")  # ✅ Store index name
        self.index = self.pc.Index(self.index_name)

    def get_closest_results(self, index_name, input_embeddings, top_k=2):
        index = self.pc.Index(index_name)
        results = index.query(
            namespace='ns1',
            vector=input_embeddings,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        return results

    def get_response(self, messages):
        messages = deepcopy(messages)

        user_message = messages[-1]['content']
        embeddings = self.embedding_client.embed_query(user_message)
        result = self.get_closest_results(self.index_name, embeddings)
        
        if result and 'matches' in result:
            source_knowledge = "\n".join([x['metadata']['text'].strip() + '\n' 
                                         for x in result['matches']])
        else:
            source_knowledge = "No relevant information found in the knowledge base."

        prompt = f"""
            Using the contexts below answer the query:
            
            Contexts:
            {source_knowledge}

            Query: {user_message}
        """

        system_prompt = """You are a customer support agent for a coffee shop called Merry's Way. You should answer every question as if you are a waiter and provide the necessary information to the user regarding their orders, menu items, and shop details."""
        
        messages[-1]['content'] = prompt
        input_messages = [{"role":"system", "content":system_prompt}] + messages[-3:]

        chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)
        output = self.postprocess(chatbot_output)
        return output

    def postprocess(self, output):
        output_dict = {
            "role": "assistant",
            "content": output,
            "memory": {
                "agent": "details_agent"
            }
        }
        return output_dict
