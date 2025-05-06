import os
import json  # Added import for json
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # Added imports for Hugging Face
import torch  # Added import for PyTorch

class AIAgent:
    """
    A general-purpose AI agent that can connect to different LLM providers.
    """
    def __init__(self, provider: str, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initializes the AI agent.

        Args:
            provider (str): The LLM provider to use ('openai', 'google', 'anthropic', 'huggingface').
            api_key (Optional[str]): The API key for the provider (if required). If None, it will attempt to
                                     read it from environment variables.
            model_name (Optional[str]): The name of the specific model to use.  Required for Hugging Face.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.client = self._get_client()  # Initialize the client

    def _get_client(self):
        """
        Retrieves the appropriate client based on the provider.
        """
        if self.provider == 'openai':
            try:
                import openai
            except ImportError:
                raise ImportError("OpenAI library not found. Please install it: pip install openai")
            if not self.api_key:
                self.api_key = os.environ.get("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError("OpenAI API key is required. Set it via the 'api_key' argument or the OPENAI_API_KEY environment variable.")
            openai.api_key = self.api_key
            return openai
        elif self.provider == 'google':
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("Google Generative AI library not found. Please install it: pip install google-generativeai")
            if not self.api_key:
                self.api_key = os.environ.get("GOOGLE_API_KEY")
                if not self.api_key:
                  raise ValueError("Google API key is required. Set it via the 'api_key' argument or the GOOGLE_API_KEY environment variable.")
            genai.configure(api_key=self.api_key)
            return genai
        elif self.provider == 'anthropic':
            try:
                import anthropic
            except ImportError:
                raise ImportError("Anthropic library not found. Please install it: pip install anthropic")
            if not self.api_key:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not self.api_key:
                    raise ValueError("Anthropic API key is required. Set it via the 'api_key' argument or the ANTHROPIC_API_KEY environment variable.")
            return anthropic.Client(api_key=self.api_key)
        elif self.provider == 'huggingface':
            if not self.model_name:
                raise ValueError("Hugging Face requires a model_name to be specified.")
            
            # Use the specified model "deepseek-ai/DeepSeek-V3-0324"
            self.model_name = "deepseek-ai/DeepSeek-V3-0324"
            
            # Option 1: Use a pipeline as a high-level helper
            print(f"Initializing Hugging Face pipeline with model '{self.model_name}'...")
            self.pipeline = pipeline("text-generation", model=self.model_name, trust_remote_code=True)
            
            # Option 2: Load model and tokenizer directly for more control
            print(f"Loading Hugging Face model and tokenizer '{self.model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)

            # Check if MPS (Metal Performance Shaders) is available
            if torch.backends.mps.is_available():
                print("Using MPS backend for model inference.")
                self.model = self.model.to("mps")  # Move model to MPS device
            else:
                print("MPS backend not available. Using CPU.")
                self.model = self.model.to("cpu")  # Fallback to CPU
            
            return self.pipeline  # Return the pipeline for high-level usage
        else:
            raise ValueError(f"Provider '{self.provider}' is not supported.  Choose 'openai', 'google', 'anthropic', or 'huggingface'.")

    def get_response(self, prompt: str, model_name: str = None, **kwargs) -> str:
        """
        Gets a response from the LLM based on the provider.

        Args:
            prompt (str): The prompt to send to the LLM.
            model_name (str, optional): The name of the specific model to use (e.g., 'gpt-3.5-turbo', 'gemini-pro', 'claude-3', 'deepseek-llm-7b-chat').
                                        If None, a default model for the provider will be used.  Required for HuggingFace.
            **kwargs:  Additional keyword arguments to pass to the provider's API.
        Returns:
            str: The response from the LLM.
        """
        if self.provider == 'openai':
            if not model_name:
                model_name = "gpt-3.5-turbo"  # Default model
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            return response.choices[0].message.content
        elif self.provider == 'google':
            if not model_name:
                model_name = 'gemini-pro'
            model = self.client.GenerativeModel(model_name)
            response = model.generate_content(prompt, **kwargs)
            return response.text
        elif self.provider == 'anthropic':
            if not model_name:
                model_name = "claude-3-opus-20240229"
            response = self.client.messages.create(
                model=model_name,
                prompt=prompt,
                max_tokens=1000,  # Or some reasonable default
                **kwargs
            )
            return response.content[0].text
        elif self.provider == 'huggingface':
            # Use the Hugging Face pipeline or model/tokenizer directly
            if not hasattr(self, 'pipeline') or not hasattr(self, 'tokenizer'):
                raise RuntimeError("Hugging Face pipeline or model/tokenizer is not initialized.")
            
            # Option 1: Use the pipeline for text generation
            response = self.pipeline([{"role": "user", "content": prompt}], **kwargs)
            return response[0]['generated_text']
            
            # Option 2: Use the model and tokenizer directly
            # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            # outputs = self.model.generate(inputs.input_ids, max_new_tokens=512, **kwargs)
            # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            raise ValueError(f"Provider '{self.provider}' is not supported.")

def load_data(file_path: str) -> Dict:
    """Loads data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def negotiate(agent: AIAgent, freelancer_data: Dict, hirer_data: Dict) -> str:
    """
    Uses the AI agent to suggest negotiation points.

    Args:
        agent (AIAgent): The initialized AI agent.
        freelancer_data (Dict): The freelancer's data.
        hirer_data (Dict): The hirer's data.

    Returns:
        str: The negotiation suggestions from the LLM.
    """
    prompt = f"""You are an expert contract negotiation assistant. Analyze the following information from a freelancer and a hiring person and suggest potential points of negotiation to reach a mutually agreeable contract.

    Freelancer Information:
    ```json
    {json.dumps(freelancer_data, indent=4)}
    ```

    Hiring Person Information:
    ```json
    {json.dumps(hirer_data, indent=4)}
    ```

    Based on this information, identify areas where the freelancer's expectations and the hiring person's offer differ. Suggest specific and actionable negotiation points for both parties to consider. Focus on hourly rate, expected hours, additional expenses, and overall budget. Provide your response in a structured format.
    """
    return agent.get_response(prompt)

if __name__ == "__main__":
    # Load data from JSON files
    try:
        freelancer_data = load_data("freelancer_data.json")
        hirer_data = load_data("hirer_data.json")
    except FileNotFoundError:
        print("Error: Make sure 'freelancer_data.json' and 'hirer_data.json' exist in the same directory.")
        print("Please run the code that generates these files first.")
        exit()

    # Initialize the AI agent with your chosen provider and API key.
    # IMPORTANT: Replace 'YOUR_PROVIDER' and 'YOUR_API_KEY' with your actual values.
    #            For example:
    #            agent = AIAgent(provider='openai', api_key='sk-...')
    #            agent = AIAgent(provider='google', api_key='YOUR_GOOGLE_API_KEY')
    #            agent = AIAgent(provider='anthropic', api_key='YOUR_ANTHROPIC_API_KEY')
    #            agent = AIAgent(provider='huggingface', model_name='TheBloke/deepseek-llm-7b-chat-AWQ') # Example
    provider = "huggingface" #  or "openai" or "google" or "anthropic"
    api_key = None #  Not generally needed for local Hugging Face models
    model_name = "TheBloke/deepseek-llm-7b-chat-AWQ"  #  REQUIRED for HuggingFace
    agent = AIAgent(provider=provider, api_key=api_key, model_name=model_name)

    # Initiate the negotiation process
    negotiation_suggestions = negotiate(agent, freelancer_data, hirer_data)
    print(f"\nNegotiation Suggestions from {provider}:")
    print(negotiation_suggestions)
