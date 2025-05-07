import json
from transformers import pipeline

class ContractNegotiatorAI:
    def __init__(self):
        self.negotiation_pipeline = None

    def setup_pipeline(self):
        """
        Sets up the HuggingFace pipeline for text generation.
        """
        try:
            self.negotiation_pipeline = pipeline("text-generation", model="gpt2", framework="pt")
            print("HuggingFace text-generation pipeline initialized successfully.")
        except Exception as e:
            print(f"Error initializing HuggingFace pipeline: {e}")

    def generate_contract(self, freelancer_data, hirer_data):
        """
        Generates a contract based on the provided freelancer and hirer data.
        """
        if not self.negotiation_pipeline:
            print("Pipeline not initialized. Please call setup_pipeline() first.")
            return None

        prompt = f"""You are an expert contract negotiator. Based on the following information, draft a contract that works for both parties:

        Freelancer Information:
        {freelancer_data}

        Hirer Information:
        {hirer_data}

        Provide a detailed and structured contract that includes terms for hourly rate, expected hours, additional expenses, and overall budget.
        """
        try:
            response = self.negotiation_pipeline(prompt, max_new_tokens=500, num_return_sequences=1)
            return response[0]["generated_text"]
        except Exception as e:
            print(f"Error generating contract: {e}")
            return None

    def negotiate(self, freelancer_data, hirer_data):
        """
        Suggests negotiation points based on freelancer and hirer data.

        Args:
            freelancer_data (dict): The freelancer's data.
            hirer_data (dict): The hirer's data.

        Returns:
            str: The negotiation suggestions.
        """
        if not self.negotiation_pipeline:
            print("Pipeline not initialized. Please call setup_pipeline() first.")
            return None

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
        try:
            response = self.negotiation_pipeline(prompt, max_new_tokens=500, num_return_sequences=1)
            return response[0]["generated_text"]
        except Exception as e:
            print(f"Error generating negotiation suggestions: {e}")
            return None

def load_data(file_path):
    """Loads data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def test_negotiation():
    """
    Tests the ContractNegotiatorAI functionality.
    """
    freelancer_data = {
        "resume": {
            "experience": [
                {"title": "Software Engineer", "company": "Tech Inc.", "years": "2020-2023", "skills": ["Python", "LLMs", "APIs"]},
                {"title": "Data Scientist", "company": "Data Corp.", "years": "2023-Present", "skills": ["Machine Learning", "Data Analysis", "Cloud Computing"]}
            ],
            "skills": ["Communication", "Problem-solving", "Negotiation"]
        },
        "expected_terms": {
            "hourly_rate": 75,
            "expected_hours_per_week": 20,
            "additional_expenses": {"software_licenses": 50, "travel": 0}
        }
    }

    hirer_data = {
        "job_description": {
            "title": "LLM Integration Specialist",
            "description": "Seeking a freelancer to integrate and fine-tune LLMs for a new application.",
            "required_skills": ["Python", "LLMs", "API Development", "Cloud Platforms"],
            "estimated_duration_weeks": 12
        },
        "expected_terms": {
            "hourly_rate": 60,
            "expected_hours_per_week": 15,
            "budget": 12000
        }
    }

    negotiator = ContractNegotiatorAI()
    negotiator.setup_pipeline()
    contract = negotiator.generate_contract(freelancer_data, hirer_data)
    if contract:
        print("\nGenerated Contract:")
        print(contract)
    else:
        print("Failed to generate contract.")

    negotiation_suggestions = negotiator.negotiate(freelancer_data, hirer_data)
    if negotiation_suggestions:
        print("\nNegotiation Suggestions:")
        print(negotiation_suggestions)
    else:
        print("Failed to generate negotiation suggestions.")

if __name__ == "__main__":
    test_negotiation()