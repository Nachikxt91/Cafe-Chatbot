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
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
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
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.index = self.pc.Index(self.index_name)

        # ✅ Add the static menu as built-in context
        self.source_knowledge_static = """
            MERRY'S WAY COFFEE SHOP - COMPLETE PRODUCT CATALOG
            
            === BEVERAGES ===
            
            ## Coffee - Drip Coffee
            - Our Old Time Diner Blend Small (8 oz): $2.00
            - Our Old Time Diner Blend Regular (16 oz): $2.50
            - Our Old Time Diner Blend Large (24 oz): $3.00
            
            ## Coffee - Organic Brewed
            - Brazilian Small (8 oz): $2.20 - It's like Carnival in a cup. Clean and smooth.
            - Brazilian Regular (16 oz): $3.00 - It's like Carnival in a cup. Clean and smooth.
            - Brazilian Large (24 oz): $3.50 - It's like Carnival in a cup. Clean and smooth.
            
            ## Coffee - Gourmet Brewed
            - Columbian Medium Roast Small (8 oz): $2.00 - A smooth cup of coffee any time of day
            - Columbian Medium Roast Regular (16 oz): $2.50 - A smooth cup of coffee any time of day
            - Columbian Medium Roast Large (24 oz): $3.00 - A smooth cup of coffee any time of day
            - Ethiopia Small (8 oz): $2.20 - A bold cup when you want that something extra
            - Ethiopia Regular (16 oz): $3.00 - A bold cup when you want that something extra
            - Ethiopia Large (24 oz): $3.50 - A bold cup when you want that something extra
            
            ## Coffee - Premium Brewed
            - Jamaican Coffee River Small (8 oz): $2.45 - Still a front runner for good premium coffee
            - Jamaican Coffee River Regular (16 oz): $3.10 - Still a front runner for good premium coffee
            - Jamaican Coffee River Large (24 oz): $3.75 - Still a front runner for good premium coffee
            
            ## Coffee - Barista Espresso
            - Espresso Shot (1.5 oz): $3.00 - You will think you are in Venice when you sip this one
            - Latte (1.5 oz): $3.75 - You will think you are in Venice when you sip this one
            - Latte Regular (3.0 oz): $4.25 - You will think you are in Venice when you sip this one
            - Cappuccino (1.5 oz): $3.75 - You will think you are in Venice when you sip this one
            - Cappuccino Large (3.0 oz): $4.25 - You will think you are in Venice when you sip this one
            - Ouro Brasileiro Shot (1.5 oz): $3.00 - From Rio
            - Ouro Brasileiro Shot Promo (16 oz): $2.10 - Ouro promo [PROMO]
            
            ## Coffee - Specialty
            - Rio Nights (6 oz): $6.00 - 2 shots of Ouro Brasilerio and pure cane sugar syrup [NEW]
            
            ## Coffee - Seasonal
            - Pumpkin Spice Latte (1.5 oz): $4.95 - Boo, its that time of year again
            - Pumpkin Spice Latte Large (3.0 oz): $5.95 - Boo, its that time of year again [PROMO]
            
            ## Tea - Herbal
            - Lemon Grass Regular (16 oz): $2.50 - You will think you are in Thailand
            - Lemon Grass Large (24 oz): $3.00 - You will think you are in Thailand
            - Peppermint Regular (16 oz): $2.50 - A cool and refreshing cup
            - Peppermint Large (24 oz): $3.00 - A cool and refreshing cup
            
            ## Tea - Green
            - Serenity Green Tea Regular (16 oz): $2.50 - Feel the stress leaving your body
            - Serenity Green Tea Large (24 oz): $3.00 - Feel the stress leaving your body
            
            ## Tea - Black
            - English Breakfast Regular (16 oz): $2.50 - The Queen's favourite cuppa in the morning
            - English Breakfast Large (24 oz): $3.00 - The Queen's favourite cuppa in the morning
            - Earl Grey Regular (16 oz): $2.50 - Tradition in a cup
            - Earl Grey Large (24 oz): $3.00 - Tradition in a cup
            
            ## Tea - Chai
            - Traditional Blend Chai Regular (16 oz): $2.50 - Sit back and think of the tropical breezes
            - Traditional Blend Chai Large (24 oz): $3.00 - Sit back and think of the tropical breezes
            - Morning Sunrise Chai Regular (16 oz): $2.50 - Face the morning after your yoga routine
            - Morning Sunrise Chai Large (24 oz): $4.00 - Face the morning after your yoga routine
            - Spicy Eye Opener Chai Regular (16 oz): $2.55 - When you need your eyes opened wide
            - Spicy Eye Opener Chai Large (24 oz): $3.10 - When you need your eyes opened wide
            
            ## Hot Chocolate
            - Dark Chocolate Regular (12 oz): $3.50 - Slightly bitter, but still very rich
            - Dark Chocolate Large (16 oz): $4.50 - Slightly bitter, but still very rich
            - Sustainably Grown Organic Regular (12 oz): $3.75 - Just pure notes of spice
            - Sustainably Grown Organic Large (16 oz): $4.75 - Just pure notes of spice
            - Snow Day Hot Chocolate (8 oz): $3.00 - Added marshmallows for the needed sugar rush
            - Happy Holidays Hot Chocolate (8 oz): $3.75 - Candy cane and hot chocolate, perfect [SEASONAL] [PROMO] [NEW]
            
            === FOOD ===
            
            ## Bakery - Pastries
            - Croissant: $3.25 - Flakey and buttery [Taxable]
            - Chocolate Croissant: $3.75 - Chocolate flakes [Taxable]
            - Almond Croissant: $3.75 - Crunch! [Taxable]
            
            ## Bakery - Scones
            - Cranberry Scone: $3.25 - Like Grandma used to make [Taxable]
            - Ginger Scone: $3.25 - Little bit of spice [Taxable]
            - Ginger Scone Promo: $2.65 - Little bit of spice [Taxable] [PROMO]
            - Oatmeal Scone: $3.00 - Grannys fav [Taxable]
            - Scottish Cream Scone: $4.50 - Old time comfort [Taxable]
            - Jumbo Savory Scone: $3.75 - Anytime, anywhere [Taxable]
            
            ## Bakery - Biscotti
            - Ginger Biscotti: $3.50 - Crunch! [Taxable]
            - Hazelnut Biscotti: $3.50 - Crunch! [Taxable]
            - Chocolate Chip Biscotti: $3.50 - Crunch! [Taxable]
            
            === WHOLE BEAN & TEAS (For Home) ===
            
            ## Coffee Beans - Organic
            - Brazilian - Organic (12 oz): $18.00 - It's like Carnival in a cup. Clean and smooth
            - Organic Decaf Blend (1 lb): $22.50 - Our blend of hand picked organic beans that have been naturally decaffinated
            
            ## Coffee Beans - House Blend
            - Our Old Time Diner Blend (12 oz): $18.00 - Our packed blend of beans that is reminiscent of the cup of coffee you used to get at a diner
            
            ## Coffee Beans - Espresso
            - Espresso Roast (1 lb): $14.75 - Our house blend for a good espresso shot
            - Primo Espresso Roast (1 lb): $20.45 - Our premium single source of hand roasted beans
            
            ## Coffee Beans - Gourmet
            - Columbian Medium Roast (1 lb): $15.00 - A smooth cup of coffee any time of day
            - Ethiopia (1 lb): $21.00 - From the home of coffee
            
            ## Coffee Beans - Premium
            - Jamaican Coffee River (1 lb): $19.75 - Ya man, it will start your day off right
            - Civet Cat (0.5 lb): $45.00 - The most expensive coffee in the world, the cats do all the work
            
            ## Coffee Beans - Green
            - Guatemalan Sustainably Grown (1 lb): $10.00 - Green beans you can roast yourself
            
            ## Loose Tea - Herbal
            - Lemon Grass (0.9 oz): $8.95 - You will think you are Thailand as you sip your cuppa
            - Peppermint (0.9 oz): $8.95 - Cool and refreshing to help calm your nerves
            
            ## Loose Tea - Black
            - English Breakfast (0.9 oz): $8.95 - The traditional cup to start your day
            - Earl Grey (0.9 oz): $8.95 - A full leaf of Orange Pekoe blended with organic oil of bergamot
            
            ## Loose Tea - Green
            - Serenity Green Tea (1 oz): $9.25 - Mountain grown and harvested at the optimal time
            
            ## Loose Tea - Chai
            - Traditional Blend Chai (0.9 oz): $8.95 - A traditional blend
            - Morning Sunrise Chai (0.9 oz): $9.50 - Fair trade and organic and has a warm finish
            - Spicy Eye Opener Chai (0.9 oz): $10.95 - A spicier blend to awaken your taste buds
            
            ## Packaged Chocolate
            - Dark Chocolate (1 lb): $6.40 - This drinking chocolate is smooth and creamy
            - Sustainably Grown Organic (1 lb): $7.60 - Certified organic containing the highest quality ingredients
            - Chili Mayan (1 lb): $13.33 - Fragrant with spices, this is the most flavourful drinking chocolate you will find
            
            === ADD-ONS ===
            
            ## Flavored Syrups
            - Caramel Syrup (per pump): $0.80 - Rich caramel taste
            - Hazelnut Syrup (per pump): $0.80 - Bursting with nutty flavour
            - Chocolate Syrup (per pump): $0.80 - Bursting with chocolate flavour
            - Sugar Free Vanilla Syrup (per pump): $0.80 - Our favorite
            
            === MERCHANDISE ===
            
            ## Branded Clothing
            - I Need My Bean! Toque: $23.00 - Keep your head bean warm [Taxable]
            - I Need My Bean! T-shirt: $28.00 - Stylish chic [Taxable]
            
            ## Branded Housewares
            - I Need My Bean! Diner Mug: $12.00 - Classic [Taxable]
            - I Need My Bean! Latte Cup: $14.00 - The cup and saucer set is the perfect way to enjoy your latte at home [Taxable]
            
            NOTES:
            - All beverages and whole bean/tea products are tax exempt
            - Food items and merchandise are taxable
            - [PROMO] = Currently on promotion
            - [NEW] = New product
            - [SEASONAL] = Seasonal availability
        """

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
        
        if result and 'matches' in result and len(result['matches']) > 0:
            dynamic_knowledge = "\n".join(
                [x['metadata']['text'].strip() + '\n' for x in result['matches']]
            )
            source_knowledge = f"{self.source_knowledge_static}\n\nRetrieved Contexts:\n{dynamic_knowledge}"
        else:
            source_knowledge = self.source_knowledge_static

        prompt = f"""
        Using the contexts below, answer the query.

        Contexts:
        {source_knowledge}

        Query: {user_message}
        """

        system_prompt = """You are a customer support agent for a coffee shop called Merry's Way.
        You should answer every question as if you are a waiter, providing detailed information about menu items,
        current promotions, and store details in a friendly tone."""

        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)
        output = self.postprocess(chatbot_output)
        return output

    def postprocess(self, output):
        return {
            "role": "assistant",
            "content": output,
            "memory": {"agent": "details_agent"}
        }
