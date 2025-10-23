from agents import (GuardAgent, 
                    ClassificationAgent,
                    DetailsAgent,
                    AgentProtocol,
                    RecommendationAgent,
                    OrderTakingAgent
                    )

from typing import Dict
import os
import pathlib

folder_path = pathlib.Path(__file__).parent.resolve()

class AgentController():
    def __init__(self):
        self.guard_agent = GuardAgent()
        self.classification_agent = ClassificationAgent()
        self.recommmendation_agent = RecommendationAgent(
                                        os.path.join(folder_path, 'recommendation_objects/apriori_recommendations.json'),
                                        os.path.join(folder_path, 'recommendation_objects/popularity_recommendation.csv')
                                    )

        self.agent_dict:Dict[str,AgentProtocol] = {
            "details_agent": DetailsAgent(),
            "recommendation_agent": self.recommmendation_agent,
            "order_taking_agent": OrderTakingAgent(self.recommmendation_agent)
        }

    def get_response(self, input):
        # Extract user input
        job_input = input["input"]
        messages = job_input["messages"]

        # Get Guard Agent's response
        guard_agent_response = self.guard_agent.get_response(messages)
        
        if guard_agent_response["memory"]["guard_decision"] == "not allowed":
            return guard_agent_response
        
        # Get classification agent's response
        classification_agent_response = self.classification_agent.get_response(messages)
        chosen_agent = classification_agent_response["memory"]["classification_decision"]

        # Get the chosen agent's response
        agent = self.agent_dict[chosen_agent]
        response = agent.get_response(messages)
        
        return response
