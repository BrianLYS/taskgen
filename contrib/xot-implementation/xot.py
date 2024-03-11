from taskgen import Function, Agent, Memory
import json
import os
from dotenv import load_dotenv
import mcts
import simulator
import policy_value_net as pv_net

# Specify the path to your .env file
dotenv_path = '../../../.env'

# Load environment variables from a .env file
load_dotenv()

# Now you can safely use the environment variable
api_key = os.getenv('OPENAI_API_KEY')

class XoT:
    def __init__(self, task_name):
        self.simulator = simulator.make(task_name)
        self.policy_value_net = pv_net.PolicyValueNet(task_name)
        self.mcts = mcts.MCTS(self.simulator, self.policy_value_net)
        self.llm = Agent(agent_name="Solution Extractor", agent_description="Extracts the final solution from the revised thoughts")
        
    def train(self, num_iterations, num_episodes):
        for i in range(num_iterations):
            experience = []
            for _ in range(num_episodes):
                experience += self.mcts.simulate()
            self.policy_value_net.train(experience)
            
    def solve(self, initial_state, num_thoughts=3, num_revisions=1):
        thoughts = self.mcts.get_thoughts(initial_state, num_thoughts)
        text_thoughts = simulator.state_to_text(thoughts)
        
        for _ in range(num_revisions):
            revised_thoughts = self.llm.run(text_thoughts)
            parsed_thoughts = simulator.text_to_state(revised_thoughts)
            thoughts = self.mcts.revise_thoughts(parsed_thoughts)
            text_thoughts = simulator.state_to_text(thoughts)
        
        solution = self.llm.run(text_thoughts)
        return solution

# Create an instance of the XoT class
xot = XoT("your_task_name")

# Train the policy/value nets
xot.train(num_iterations=10, num_episodes=100)

# Define the initial state for the task
initial_state = "..."

# Solve the task using the trained XoT
solution = xot.solve(initial_state, num_thoughts=3, num_revisions=2)

# Print the solution
print("Solution:", solution)