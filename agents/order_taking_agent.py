import os
import json
from .utils import get_chatbot_response, double_check_json_output
from groq import Groq
from copy import deepcopy
from dotenv import load_dotenv

load_dotenv()

class OrderTakingAgent():
    def __init__(self, recommendation_agent):
        self.client = Groq(
            api_key=os.getenv('GROQ_API_KEY')
        )
        self.model_name = os.getenv("MODEL_NAME")

        self.recommendation_agent = recommendation_agent

    def get_response(self, messages):
        messages = deepcopy(messages)

        system_prompt = """
            You are a customer support Bot for a coffee shop called "Merry's way"

            here is the menu for this coffee shop.

            Cappuccino - $4.50
            Jumbo Savory Scone - $3.25
            Latte - $4.75
            Chocolate Chip Biscotti - $2.50
            Espresso shot - $2.00
            Hazelnut Biscotti - $2.75
            Chocolate Croissant - $3.75
            Dark chocolate (Drinking Chocolate) - $5.00
            Cranberry Scone - $3.50
            Croissant - $3.25
            Almond Croissant - $4.00
            Ginger Biscotti - $2.50
            Oatmeal Scone - $3.25
            Ginger Scone - $3.50
            Chocolate syrup - $1.50
            Hazelnut syrup - $1.50
            Carmel syrup - $1.50
            Sugar Free Vanilla syrup - $1.50
            Dark chocolate (Packaged Chocolate) - $3.00

            Things to NOT DO:
            * DON't ask how to pay by cash or Card.
            * Don't tell the user to go to the counter
            * Don't tell the user to go to place to get the order

            You're task is as follows:
            1. Take the User's Order
            2. Validate that all their items are in the menu
            3. if an item is not in the menu let the user and repeat back the remaining valid order
            4. Ask them if they need anything else.
            5. If they do then repeat starting from step 3
            6. If they don't want anything else. CRITICAL: You MUST use the COMPLETE "order" array from the memory section without modification. Do NOT create a new order array. Make sure to:
                1. list down ALL the items from the order array and their prices
                2. calculate the total by adding ALL prices from the order array
                3. Thank the user for the order and close the conversation with no more questions

            IMPORTANT: The user message will contain a section called memory with:
            - "order": An array containing ALL items ordered so far - YOU MUST PRESERVE THIS COMPLETE ARRAY
            - "step number": The current step
            
            When finalizing the order (step 6), copy the ENTIRE order array from memory WITHOUT CHANGES.
            Do NOT create a new order array. Do NOT remove any items.
            
            produce the following output without any additions, not a single letter outside of the structure bellow.
            Your output should be in a structured json format like so:
            {
            "chain of thought": "First, check if there is an existing order array in memory. If yes and user wants to finalize, COPY IT EXACTLY. Write down the step number and user intent. If user is done ordering, list ALL items from the existing order array and calculate the complete total.",
            "step number": "Determine which task you are on (1-6)",
            "order": "If an order exists in memory, COPY IT EXACTLY. If adding new items, append to the existing order. This is a list of jsons: [{'item':'item name', 'quantity': number, 'price':individual item price * quantity}]",
            "response": "Your response to the user"
            }
        """

        last_order_taking_status = ""
        asked_recommendation_before=False
        for message_index in range(len(messages)-1, -1, -1):  # Fixed: include index 0
            message = messages[message_index]
            
            agent_name = message.get("memory",{}).get("agent","")
            if message["role"] == "assistant" and agent_name=="order_taking_agent":
                step_number = message['memory']['step_number']
                order = message["memory"]["order"]
                asked_recommendation_before= message["memory"]["asked_recommendation_before"]
                last_order_taking_status = f"""
                step number: {step_number}
                order: {order}
                """
                break  # Added: Stop after finding the most recent order state
        
        messages[-1]['content'] = last_order_taking_status + " \n" + messages[-1]['content']
        input_messages = [{"role":"system","content":system_prompt}] + messages

        chatbot_response = get_chatbot_response(self.client, self.model_name, input_messages)
        chatbot_response = double_check_json_output(self.client, self.model_name, chatbot_response)

        output = self.postprocess(chatbot_response, messages, asked_recommendation_before)

        return output
    
    def postprocess(self, output, messages, asked_recommendation_before):
        output = json.loads(output)

        if type(output['order']) == str:
            output['order'] = json.loads(output['order'])

        response = output["response"]

        if not asked_recommendation_before and len(output["order"]) > 0:
            recommendation_output = self.recommendation_agent.get_recommendations_from_order(messages,output["order"])
            response = recommendation_output['content']
            asked_recommendation_before = True

        dict_output = {
            "role":"assistant",
            "content":response,
            "memory":{
                "agent":"order_taking_agent",
                "step_number":output["step number"],
                "asked_recommendation_before": asked_recommendation_before,
                "order":output["order"],
            }
        }

        return dict_output