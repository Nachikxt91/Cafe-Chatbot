from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from groq import Groq
import os
import dotenv
import json
from copy import deepcopy
from .utils import get_chatbot_response, double_check_json_output

dotenv.load_dotenv()

class GuardAgent:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv('GROQ_API_KEY')
        )
        self.model_name = os.getenv("MODEL_NAME")

    def get_response(self, messages):
        messages = deepcopy(messages)

        system_prompt="""
            You are a security guard AI assistant for a coffee shop application.
            Your ONLY task is to determine if the user's request is allowed or not allowed.
            
            The user is allowed to:
            1. Ask questions about the coffee shop, like location, working hours, menu items
            2. Ask questions about menu items, ingredients, prices, or details
            3. Make an order
            4. Ask about recommendations of what to buy
            
            The user is NOT allowed to:
            1. Ask questions about anything else other than our coffee shop
            2. Ask questions about the staff or how to make a certain menu item
            
            IMPORTANT: Your job is ONLY to validate if the request is allowed. DO NOT answer the user's question.
            
            Your output should be in a structured JSON format:
            {
            "chain_of_thoughts": "Analyze if this request is related to the coffee shop or not.",
            "decision": "allowed" or "not allowed",
            "message": "If 'not allowed': write 'Sorry, I can't help with that. Can I help you with your order?'. If 'allowed': write an empty string ''"
            }
        """

        input_messages = [{"role":"system", "content": system_prompt}] + messages[-3:]

        chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)
        chatbot_output = double_check_json_output(self.client, self.model_name, chatbot_output)
        output = self.postprocess(chatbot_output)

        return output


    def postprocess(self, output):
        output = json.loads(output)

        dict_output = {
            "role":"assistant",
            "content": output["message"],
            "memory": {
                "agent": "guard_agent",
                "guard_decision": output["decision"],
            }
        }

        return dict_output