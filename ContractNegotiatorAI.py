import json
from transformers import pipeline, GPT2Tokenizer
import logging  # Import the logging module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ContractNegotiatorAI:
    def __init__(self):
        """
        Initializes the ContractNegotiatorAI.
        """
        self.negotiation_pipeline = None
        self.tokenizer = None  # Add tokenizer

    def setup_pipeline(self):
        """
        Sets up the HuggingFace pipeline for text generation.
        """
        try:
            # Use a specific model and handle potential errors
            model_name = "gpt2"
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)  # Load tokenizer
            self.negotiation_pipeline = pipeline("text-generation", model=model_name)
            logging.info("HuggingFace text-generation pipeline initialized successfully with gpt2.")
        except Exception as e:
            logging.error(f"Failed to initialize HuggingFace pipeline: {e}")
            self.negotiation_pipeline = None  # Ensure pipeline is None on failure
            self.tokenizer = None
            return  # Exit the setup method on failure

    def truncate_prompt(self, prompt, max_length=1024):
        """
        Truncates the input prompt to ensure it does not exceed the model's maximum sequence length.
        Uses the tokenizer loaded in setup_pipeline.

        Args:
            prompt (str): The input prompt.
            max_length (int): The maximum sequence length allowed by the model.

        Returns:
            str: The truncated prompt.
        """
        if not self.tokenizer:
            logging.error("Tokenizer not initialized.  Cannot truncate prompt.")
            return prompt  # Return original prompt to avoid crashing

        tokenized_prompt = self.tokenizer.encode(prompt, truncation=True, max_length=max_length)
        return self.tokenizer.decode(tokenized_prompt)

    def generate_contract(self, freelancer_data, hirer_data):
        """
        Generates a contract based on the provided freelancer and hirer data using the pipeline.
        Focuses on SOW, hours, and rate.
        """
        if not self.negotiation_pipeline:
            logging.error("Pipeline not initialized. Please call setup_pipeline() first.")
            return None

        # Extract relevant information for the contract
        job_title = hirer_data['job_description']['title']
        job_description = hirer_data['job_description']['description']
        required_skills = hirer_data['job_description']['required_skills']
        estimated_duration_weeks = hirer_data['job_description']['estimated_duration_weeks']
        hourly_rate = hirer_data['expected_terms']['hourly_rate']
        expected_hours_per_week = hirer_data['expected_terms']['expected_hours_per_week']
        budget = hirer_data['expected_terms']['budget']  # added
        freelancer_name = freelancer_data['profile']['name']  # added
        freelancer_title = freelancer_data['profile']['title']  # added

        prompt = f"""You are an expert contract writer. Based on the following information, draft a contract that works for both parties:

        Freelancer Name: {freelancer_name}
        Freelancer Title: {freelancer_title}

        Hirer Information:
        Job Title: {job_title}
        Job Description: {job_description}
        Required Skills: {required_skills}
        Estimated Duration: {estimated_duration_weeks} weeks

        Freelancer Proposed Terms:
        Hourly Rate: {freelancer_data['expected_terms']['hourly_rate']}
        Expected Hours Per Week: {freelancer_data['expected_terms']['expected_hours_per_week']}
        Additional Expenses: {freelancer_data['expected_terms']['additional_expenses']}

        Hirer Proposed Terms:
        Hourly Rate: {hourly_rate}
        Expected Hours Per Week: {expected_hours_per_week}
        Budget: {budget}

        Provide a detailed and structured contract that includes the following sections:

        1.  **Scope of Work (SOW):** A clear description of the services to be provided by the freelancer, including specific tasks, deliverables, and timelines.  Incorporate the Job Description and Required Skills.
        2.  **Terms of Service:**
            * Hourly Rate: The agreed-upon hourly rate.
            * Expected Hours: The expected number of hours per week.
            * Project Duration: The estimated duration of the project.
            * Payment Terms: How and when the freelancer will be paid.
            * Expenses: Reimbursable expenses, if any.
        3.  **Acceptance:** A section for both parties to sign and date.
        """

        # Truncate the prompt to avoid exceeding the model's maximum sequence length
        truncated_prompt = self.truncate_prompt(prompt)  # Use the class's tokenizer
        if not truncated_prompt:
            return None  # handle error in truncation

        try:
            generated_text = self.negotiation_pipeline(truncated_prompt, max_new_tokens=800,
                                                      num_return_sequences=1)[0]["generated_text"]
        except Exception as e:
            logging.error(f"Error generating contract: {e}")
            return None

        # Add boilerplate text
        boilerplate_text = """
        \n\n------------------------------------------------------------------------
        **General Terms and Conditions**

        1.  **Independent Contractor Relationship:** The Freelancer is an independent contractor and not an employee of the Client.  This agreement does not create a partnership, joint venture, or agency relationship between the parties.

        2.  **Ownership of Work Product:** The Client shall own all right, title, and interest in and to the work product resulting from the services performed by the Freelancer under this Agreement, including all intellectual property rights.

        3.  **Confidentiality:** The Freelancer agrees to hold all confidential information of the Client in strict confidence and not to disclose such information to any third party without the Client's prior written consent.

        4.  **Termination:** This Agreement may be terminated by either party upon [Number] days written notice to the other party.  In the event of termination, the Freelancer shall be paid for all services performed up to the date of termination.

        5.  **Indemnification:** The Freelancer agrees to indemnify and hold the Client harmless from any and all claims, losses, damages, liabilities, costs, and expenses (including reasonable attorneys' fees) arising out of or relating to the Freelancer's performance of services under this Agreement.

        6.  **Governing Law:** This Agreement shall be governed by and construed in accordance with the laws of [State/Country].

        7.  **Entire Agreement:** This Agreement constitutes the entire agreement between the parties and supersedes all prior agreements and understandings, whether written or oral, relating to the subject matter of this Agreement.
        """

        return generated_text + boilerplate_text

    def negotiate(self, freelancer_data, hirer_data):
        """
        Suggests negotiation points based on freelancer and hirer data.  Implements rule-based negotiation.

        Args:
            freelancer_data (dict): The freelancer's data.
            hirer_data (dict): The hirer's data.

        Returns:
            dict: The agreed-upon terms, or None if no agreement is reached.
        """
        if not self.negotiation_pipeline:
            logging.error("Pipeline not initialized. Please call setup_pipeline() first.")
            return None

        # 1. Extract relevant information
        freelancer_rate = freelancer_data['expected_terms']['hourly_rate']
        freelancer_min_rate = freelancer_data['expected_terms'].get('minimum_acceptable_rate',
                                                                  freelancer_rate - 10)  # Provide a default
        freelancer_hours = freelancer_data['expected_terms']['expected_hours_per_week']
        freelancer_max_hours = freelancer_data['expected_terms'].get('maximum_weekly_capacity',
                                                                    freelancer_hours + 10)  # Provide a default
        hirer_rate = hirer_data['expected_terms']['hourly_rate']
        hirer_budget = hirer_data['budget']['maximum_hourly_rate']
        hirer_hours = hirer_data['expected_terms']['expected_hours_per_week']
        estimated_duration = hirer_data['job_description']['estimated_duration_weeks']
        freelancer_preferred_duration = freelancer_data['expected_terms'].get(
            'preferred_project_length_weeks', estimated_duration)  # added

        # 2. Initialize negotiation state
        negotiation_params = {
            "hourly_rate": {"freelancer": freelancer_rate, "hirer": hirer_rate},
            "hours_per_week": {"freelancer": freelancer_hours, "hirer": hirer_hours},
            "project_duration": {"freelancer": freelancer_preferred_duration,
                                 "hirer": estimated_duration},  # added
        }
        offer_history = []
        MAX_ROUNDS = 5  # Limit negotiation rounds

        # 3. Negotiation loop
        for round_num in range(MAX_ROUNDS):
            logging.info(f"--- Round {round_num + 1} ---")
            offer_history.append(negotiation_params.copy())

            # Check for agreement on key terms
            rate_agreement = freelancer_min_rate <= negotiation_params["hourly_rate"]["hirer"] <= hirer_budget
            hours_agreement = freelancer_hours <= negotiation_params["hours_per_week"]["hirer"] <= freelancer_max_hours
            duration_agreement = negotiation_params["project_duration"]["freelancer"] == \
                                  negotiation_params["project_duration"]["hirer"]

            if rate_agreement and hours_agreement and duration_agreement:
                logging.info("Agreement reached on hourly rate, hours per week, and project duration.")
                return {
                    "hourly_rate": negotiation_params["hourly_rate"]["hirer"],
                    "hours_per_week": negotiation_params["hours_per_week"]["hirer"],
                    "project_duration": negotiation_params["project_duration"]["hirer"],
                }

            # 4. Generate counter-offers (simplified rules)
            if not rate_agreement:
                if negotiation_params["hourly_rate"]["freelancer"] > negotiation_params["hourly_rate"]["hirer"]:
                    negotiation_params["hourly_rate"]["freelancer"] = max(
                        freelancer_min_rate, negotiation_params["hourly_rate"]["freelancer"] - 5
                    )  # Freelancer concedes
                    logging.info(
                        f"Freelancer counter-offer: hourly_rate={negotiation_params['hourly_rate']['freelancer']}")
                elif negotiation_params["hourly_rate"]["hirer"] < hirer_budget:
                    negotiation_params["hourly_rate"]["hirer"] = min(
                        hirer_budget, negotiation_params["hourly_rate"]["hirer"] + 5
                    )  # Hirer concedes
                    logging.info(f"Hirer counter-offer: hourly_rate={negotiation_params['hourly_rate']['hirer']}")
                else:
                    logging.info("Agreement cannot be reached on hourly rate.")
                    return None  # No agreement

            if not hours_agreement:
                if negotiation_params["hours_per_week"]["freelancer"] > \
                        negotiation_params["hours_per_week"]["hirer"]:
                    negotiation_params["hours_per_week"]["freelancer"] = max(
                        freelancer_hours, negotiation_params["hours_per_week"]["freelancer"] - 2
                    )
                    logging.info(
                        f"Freelancer counter-offer: hours_per_week={negotiation_params['hours_per_week']['freelancer']}"
                    )
                else:
                    negotiation_params["hours_per_week"]["hirer"] = min(
                        freelancer_max_hours, negotiation_params["hours_per_week"]["hirer"] + 2
                    )
                    logging.info(
                        f"Hirer counter-offer: hours_per_week={negotiation_params['hours_per_week']['hirer']}"
                    )

            if not duration_agreement:
                if isinstance(negotiation_params["project_duration"]["freelancer"],
                              list):  # if freelancer provides a list of preferred durations
                    if estimated_duration in negotiation_params["project_duration"]["freelancer"]:
                        negotiation_params["project_duration"]["freelancer"] = estimated_duration
                        logging.info(f"Freelancer agrees to project duration of {estimated_duration} weeks")
                else:
                    negotiation_params["project_duration"]["freelancer"] = estimated_duration
                    logging.info(f"Freelancer agrees to project duration of {estimated_duration} weeks")
        logging.info("Negotiation failed: Maximum rounds reached.")
        return None  # No agreement reached


def test_negotiation():
    """
    Tests the ContractNegotiatorAI functionality using the provided JSON data.
    """
    freelancer_data = {
        "profile": {
            "name": "Alice Smith",
            "title": "Senior LLM Engineer",
        },
        "resume": {
            "experience": [
                {"title": "Software Engineer", "company": "Tech Inc.", "years": "2020-2023",
                 "skills": ["Python", "LLMs", "APIs"]},
                {"title": "Data Scientist", "company": "Data Corp.", "years": "2023-Present",
                 "skills": ["Machine Learning", "Data Analysis", "Cloud Computing"]}
            ],
            "skills": ["Communication", "Problem-solving", "Negotiation"]
        },
        "expected_terms": {
            "hourly_rate": 75,
            "expected_hours_per_week": 20,
            "additional_expenses": {"software_licenses": 50, "travel": 0},
            # "minimum_acceptable_rate": 65,
            # "maximum_weekly_capacity": 25,
            # "preferred_project_length_weeks": [8, 16],
            "payment_terms": "Net 30"
        }
    }

    hirer_data = {
        "project": {
            "name": "LLM Integration for Marketing Platform",
            "description": "...",
            "priority": "High"
        },
        "job_description": {
            "title": "LLM Integration Specialist",
            "description": "Seeking a freelancer to integrate and fine-tune LLMs for a new application.",
            "required_skills": ["Python", "LLMs", "API Development", "Cloud Platforms"],
            "estimated_duration_weeks": 12
        },
        "expected_terms": {
            "hourly_rate": 60,
            "expected_hours_per_week": 15,
            "budget": 12000,
            # "flexibility_on_hours": True,
            # "payment_terms": ["Net 30", "Net 45"]
        },
        "budget": {
            "total": 12000,
            "maximum_hourly_rate": 70
        }
    }

    negotiator = ContractNegotiatorAI()
    negotiator.setup_pipeline()  # Initialize the pipeline

    # Generate and Print Contract
    contract = negotiator.generate_contract(freelancer_data, hirer_data)
    if contract:
        print("\nGenerated Contract:")
        print(contract)
    else:
        print("Failed to generate contract.")
        
if __name__ == "__main__":
    test_negotiation()
