import os
import json  # Added import for json
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # Added imports for Hugging Face
import torch  # Added import for PyTorch
from openai import OpenAI  # Added import for OpenAI client
from pydantic import BaseModel, Field  # Added Field for explicit schema definitions

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

    def get_response(self, prompt: str, model_name: str = None, structured: bool = False, output_format: Optional[BaseModel] = None, **kwargs) -> str:
        """
        Gets a response from the LLM based on the provider.

        Args:
            prompt (str): The prompt to send to the LLM.
            model_name (str, optional): The name of the specific model to use (e.g., 'gpt-3.5-turbo', 'gpt-4.1').
                                        If None, a default model for the provider will be used.
            structured (bool): Whether to parse the response into a structured format (e.g., SOW).
            output_format (Optional[BaseModel]): The Pydantic model to use for structured parsing.
            **kwargs: Additional keyword arguments to pass to the provider's API.
        Returns:
            str or BaseModel: The response from the LLM, either as plain text or a structured object.
        """
        if self.provider == 'openai':
            if not model_name:
                model_name = "gpt-4.1"  # Default model for OpenAI
            if structured and output_format:
                # Temporarily disable structured response parsing
                # response = self.client.responses.parse(
                #     model=model_name,
                #     input=[
                #         {"role": "system", "content": "Generate a structured SOW based on the input."},
                #         {"role": "user", "content": prompt},
                #     ],
                #     text_format=output_format,
                #     **{k: v for k, v in kwargs.items() if k != "max_tokens"}  # Exclude max_tokens
                # )
                # return response.output_parsed
                raise NotImplementedError("Structured response parsing is temporarily disabled.")
            else:
                # Use standard text generation (remove max_tokens)
                response = self.client.responses.create(
                    model=model_name,
                    input=[
                        {"role": "user", "content": prompt}
                    ],
                    **{k: v for k, v in kwargs.items() if k != "max_tokens"}  # Exclude max_tokens
                )
                return response.output_text
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

class SOW(BaseModel):
    """Defines a structured response format for a Statement of Work (SOW)."""
    job_type: str = Field(..., description="The specific job type.")
    persona: str = Field(..., description="The name of the user persona.")
    sow_title: str = Field(..., description="A brief, descriptive title for the SOW.")
    introduction: str = Field(..., description="A short paragraph providing context.")
    objectives: list[str] = Field(..., description="An array of clearly stated, measurable objectives.")
    scope_of_work: list[str] = Field(..., description="An array of specific tasks and deliverables included.")
    deliverables: list[str] = Field(..., description="An array of tangible outputs.")
    timeline_milestones: dict[str, str] = Field(..., description="An object where keys are milestone names and values are estimated completion dates (YYYY-MM-DD).")
    acceptance_criteria: list[str] = Field(..., description="An array of criteria for deliverable acceptance.")
    reporting_communication: str = Field(..., description="A string describing reporting frequency and methods.")
    assumptions: list[str] = Field(..., description="An array of critical assumptions.")
    exclusions: list[str] = Field(..., description="An array of tasks or deliverables NOT included.")

    class Config:
        json_schema_extra = {  # Updated from schema_extra to json_schema_extra
            "required": [
                "job_type",
                "persona",
                "sow_title",
                "introduction",
                "objectives",
                "scope_of_work",
                "deliverables",
                "timeline_milestones",
                "acceptance_criteria",
                "reporting_communication",
                "assumptions",
                "exclusions"
            ]
        }

def load_data(file_path: str) -> Dict:
    """Loads data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    print("AIAgent module is ready for use.")
