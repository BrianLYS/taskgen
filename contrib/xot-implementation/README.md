# Implementation of the Everything of Thought paper
This script provides an example of implementing XoT.

## Features
The key steps are:
- Train the policy/value nets using MCTS simulations
- For each task instance, run MCTS to get initial thoughts
- Iteratively revise thoughts via LLM and MCTS
- Extract final solution from revised thoughts using LLM

## Prerequisites
- Python 3.10
- taskgen library
- python-dotenv library
- OpenAI API key
## Setup
Install the required dependencies:

```bash
pip install -r requirements.txt
```
Create a `.env` file in the project root directory and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage


## Customization
