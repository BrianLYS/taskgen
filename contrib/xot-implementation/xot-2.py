from taskgen import Function, Agent, Memory
import os
from dotenv import load_dotenv
import numpy as np
import polars as pl

# Specify the path to your .env file
dotenv_path = '../../../.env'

# Load environment variables from a .env file
load_dotenv()

# Now you can safely use the environment variable
api_key = os.getenv('OPENAI_API_KEY')

class XoT:
    def __init__(self, task_name):
        self.simulator = make(task_name)
        self.policy_value_net = PolicyValueNet(task_name)
        self.mcts = MCTS(self.simulator, self.policy_value_net)
        self.llm = Agent(agent_name="Solution Extractor", agent_description="Extracts the final solution from the revised thoughts")
        
        # Define the function to extract the solution from thoughts
        extract_solution_fn = Function(
            fn_name="extract_solution",
            fn_description="Extract the final solution from the revised thoughts",
            output_format={"solution": "The final solution extracted from the revised thoughts"}
        )
        
        # Assign the function to the LLM agent
        self.llm.assign_functions(extract_solution_fn)
        
    def train(self, num_iterations, num_episodes):
        for i in range(num_iterations):
            print(f"Iteration: {i+1}/{num_iterations}")
            experience = []
            for j in range(num_episodes):
                print(f"  Episode: {j+1}/{num_episodes}")
                exp = self.mcts.simulate()
                if exp:
                    experience.extend(exp)
            if experience:
                self.policy_value_net.train(experience)
            else:
                print("No valid experience to train on in this iteration.")
            
    def solve(self, initial_state, num_thoughts=3, num_revisions=1):
        print("Initial state:")
        self.print_board(initial_state['board'])
        thoughts = self.mcts.get_thoughts(initial_state, num_thoughts)
        print(f"Thoughts (before revision):")
        for thought in thoughts:
            # Corrected to pass the list (board state) directly
            self.print_board(thought)
        text_thoughts = TicTacToeSimulator.state_to_text(thoughts)

        for _ in range(num_revisions):
            revised_thoughts = self.llm.run(task=f"Revise the thoughts:\n{text_thoughts}")
            # Make sure the parsing returns the correct format if needed
            parsed_thoughts = TicTacToeSimulator.text_to_state(revised_thoughts)
            thoughts = self.mcts.revise_thoughts(parsed_thoughts)
            print(f"Revised thoughts:")
            for thought in thoughts:
                # Again, corrected to use the direct board state
                self.print_board(thought)
            text_thoughts = TicTacToeSimulator.state_to_text(thoughts)

        solution = self.llm.run(task=f"Extract the final solution from the revised thoughts:\n{text_thoughts}")
        print("Final solution:")
        print(solution["solution"])
        return solution["solution"]

    def print_board(self, board):
        for row in board:
            print(" ".join(row))
        print()


# Monte Carlo Tree Search (MCTS)
class MCTS:
    def __init__(self, simulator, policy_value_net, num_simulations=100):
        self.simulator = simulator
        self.policy_value_net = policy_value_net
        self.num_simulations = num_simulations
        
    def simulate(self):
        experience = []
        state = self.simulator.reset()
        print("Initial state:")
        self.print_board(state['board'])
        while not self.simulator.is_terminal():
            action_probs = self.policy_value_net.predict(state)
            valid_actions = [i for i, cell in enumerate(self.simulator.board) if cell == '']
            if np.random.rand() < 0.1:  # 10% exploration
                if valid_actions:
                    action = np.random.choice(valid_actions)
                else:
                    break  # No valid actions available, end the simulation
            else:  # 90% exploitation
                if valid_actions:
                    valid_action_probs = action_probs[valid_actions]
                    action_idx = np.argmax(valid_action_probs)
                    action = valid_actions[action_idx]
                else:
                    break  # No valid actions available, end the simulation
            next_state, reward, done = self.simulator.step(action)
            print(f"Action taken: {action}")
            print("Next state:")
            self.print_board(next_state['board'])
            experience.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                break  # Game is over, end the simulation
        print("Game over.")
        if experience:
            return experience
        else:
            return None
    
    def print_board(self, board):
        for row in board:
            print(" ".join(row))
        print()
    
    def get_thoughts(self, state, num_thoughts):
        thoughts = []
        for _ in range(num_thoughts):
            thought = self.search(state)
            thoughts.append(thought['board'])  # Pass only the board state
        return thoughts

    def revise_thoughts(self, thoughts):
        revised_thoughts = []
        for thought in thoughts:
            revised_thought = self.search(thought)
            revised_thoughts.append(revised_thought)
        return revised_thoughts
    
    def search(self, state):
        for _ in range(self.num_simulations):
            self.simulator.set_state(state)
            self.simulate()
        return self.simulator.get_state()

# Policy/Value Network
class PolicyValueNet:
    def __init__(self, task_name):
        self.task_name = task_name
        # Initialize the neural network architecture
        self.weights = np.random.randn(9, 1)  # Example: Random weights for Tic-Tac-Toe
        
    def predict(self, state):
        # Forward pass through the neural network to get action probabilities
        board = state['board']
        flat_board = [item for sublist in board for item in sublist]  # Flatten the board
        x = np.array([1 if cell == 'X' else (-1 if cell == 'O' else 0) for cell in flat_board])
        x = x.reshape(1, -1)  # Reshape x to (1, 9)
        action_probs = np.dot(x, self.weights)[0]  # Perform matrix multiplication and extract the result
        action_probs = np.exp(action_probs)  # Apply exponential for non-negative probabilities
        action_probs /= action_probs.sum()  # Normalize probabilities
        return action_probs
    
    def train(self, experience, learning_rate=0.01, num_epochs=10):
        if not experience:
            return
        x = []
        y = []
        for state, action, reward, next_state, done in experience:
            board = state['board']
            flat_board = [item for sublist in board for item in sublist]  # Flatten the board
            x.append(np.array([1 if cell == 'X' else (-1 if cell == 'O' else 0) for cell in flat_board]))
            y.append(action)
        x = np.array(x)
        y = np.array(y)

        for _ in range(num_epochs):
            # Forward pass
            action_probs = np.dot(x, self.weights)
            action_probs = np.exp(action_probs)
            action_probs /= action_probs.sum(axis=1, keepdims=True)

            # Backward pass
            grad = np.dot(x.T, (action_probs - np.eye(9)[y]))
            self.weights -= learning_rate * grad

# Simulator
class TicTacToeSimulator:
    def __init__(self):
        self.board = [['', '', ''], ['', '', ''], ['', '', '']]
        self.player = 'X'
        
    def reset(self):
        self.board = [['', '', ''], ['', '', ''], ['', '', '']]
        self.player = 'X'
        return {'board': self.board, 'player': self.player}
    
    def step(self, action):
        row, col = action // 3, action % 3
        if self.board[row][col] != '':  # Check if the cell is already taken
            return {'board': self.board, 'player': self.player}, 0, False  # No change, no reward, not done
        self.board[row][col] = self.player
        self.player = 'O' if self.player == 'X' else 'X'
        done = self.is_terminal()
        reward = 1 if self.get_winner() == 'X' else 0
        next_state = {'board': self.board, 'player': self.player}
        return next_state, reward, done
    
    def set_state(self, state):
        self.board = state['board']
        self.player = state.get('player', 'X')  # Provide a default value if 'player' key is missing
        
    def get_state(self):
        return {'board': self.board, 'player': self.player}
    
    @staticmethod
    def state_to_text(states):
        """
        Converts a list of board states to a text representation.
        
        Each state is expected to be a board (a list of lists, with each inner list representing a row).
        """
        text_representation = ""
        for state in states:
            for row in state:
                text_representation += "".join(['_' if cell == '' else cell for cell in row]) + "\n"
            text_representation += "\n---\n"  # Separator between different board states
        return text_representation

    @staticmethod
    def text_to_state(text):
        board = [list(row) for row in text.split("\n") if row]
        return {'board': board, 'player': 'X'}  # Assuming 'X' starts after revising thoughts


    def is_terminal(self):
        # Check rows
        for row in self.board:
            if row.count(row[0]) == len(row) and row[0] != '':
                return True

        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] and self.board[0][col] != '':
                return True

        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != '':
            return True

        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != '':
            return True

        # Check if board is full
        if all(cell != '' for row in self.board for cell in row):
            return True

        return False
    
    def get_winner(self):
        # Check rows
        for row in self.board:
            if row.count(row[0]) == len(row) and row[0] != '':
                return row[0]
        
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] and self.board[0][col] != '':
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != '':
            return self.board[0][0]
        
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != '':
            return self.board[0][2]
        
        return None

def make(task_name):
    if task_name == "tictactoe":
        return TicTacToeSimulator()
    else:
        raise ValueError(f"Unknown task: {task_name}")

# Create an instance of the XoT class
xot = XoT("tictactoe")

# Train the policy/value nets
xot.train(num_iterations=100, num_episodes=1000)

# Define the initial state for the task
initial_state = {'board': [['', '', ''], ['', '', ''], ['', '', '']], 'player': 'X'}

# Solve the task using the trained XoT
solution = xot.solve(initial_state, num_thoughts=3, num_revisions=2)

# Print the solution
print("Solution:", solution)