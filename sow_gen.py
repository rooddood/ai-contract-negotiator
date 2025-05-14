import os
from AIAgent import AIAgent, SOW  # Import SOW class

def load_prompt(file_path: str) -> str:
    """Loads the SOW generation prompt from a text file."""
    with open(file_path, 'r') as f:
        return f.read()

def generate_sow():
    """Generates a Statement of Work (SOW) using an AI agent."""
    # Load the SOW generation prompt
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "sow_generation_prompt.txt")
    prompt = load_prompt(prompt_path)

    # Initialize the AI agent with OpenAI as the provider
    agent = AIAgent(provider="openai", api_key=os.environ.get("OPENAI_API_KEY"))

    # Call the LLM with the loaded prompt
    response = agent.get_response(
        prompt=prompt,
        model_name="gpt-4.1"
    )

    # Print the structured response (or handle it as needed)
    print("Generated SOW:")
    print(response.json(indent=4))

if __name__ == "__main__":
    generate_sow()
