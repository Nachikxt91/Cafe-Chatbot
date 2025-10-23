from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from groq import Groq
import os
import dotenv
import json
from copy import deepcopy
from .utils import get_chatbot_response, double_check_json_output


dotenv.load_dotenv()


class ClassificationAgent():
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv('GROQ_API_KEY')
        )
        self.model_name = os.getenv("MODEL_NAME")


    def get_response(self, messages):
        messages = deepcopy(messages)


        system_prompt="""
            You are an intent classification AI assistant for a coffee shop application.
            Your task is to determine which specialized agent should handle the user's request.
            
            You have 3 agents to choose from:
            
            1. details_agent: Use ONLY when the user is:
            - Asking about SPECIFIC details: "What is the price of Latte?"
            - Asking about ingredients: "What's in the Croissant?"
            - Asking about shop info: "What are your hours?" "Where are you located?"
            - Asking to see the menu: "What do you have?" "Show me the menu"
            
            2. order_taking_agent: Use ONLY when the user is:
            - Explicitly stating they want to order: "I want to order a Latte"
            - Using order phrases: "I'll have...", "Can I get...", "Give me..."
            - Confirming or modifying an order
            - User signals completion: "that's all", "that will be all"
            
            3. recommendation_agent: Use when the user is asking for:
            - Suggestions: "what can I order?", "what should I get?"
            - Pairings: "what can I order with Latte?", "what goes well with..."
            - Recommendations: "what do you recommend?", "what's good?"
            - General browsing: "what can I order?" (without specifics)
            
            Your output must be in this exact JSON format (NO OTHER KEYS ALLOWED):
            {
            "chain_of_thought": "Brief reasoning",
            "classification_decision": "Write ONLY one: details_agent, order_taking_agent, or recommendation_agent",
            "message": ""
            }
        """


        input_messages = [{"role":"system", "content": system_prompt}] + messages[-3:]


        chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)
        chatbot_output = double_check_json_output(self.client, self.model_name, chatbot_output)
        output = self.postprocess(chatbot_output)


        return output


    def postprocess(self, output):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return {
                'role': 'assistant',
                'content': '',
                'memory': {
                    'agent': 'classification_agent',
                    'classification_decision': 'order_taking_agent'
                }
            }
        
        classification = output.get("classification_decision", output.get("decision", "order_taking_agent"))
        
        dict_output = {
            'role': 'assistant',
            'content': output.get("message", ""),
            'memory': {
                'agent': 'classification_agent',
                'classification_decision': classification
            }
        }
        
        return dict_output
