import json
from transformers import pipeline, GPT2Tokenizer
import logging  # Import the logging module
import os  # Add import for os
from HuggingFaceAI import HuggingFaceAI  # Import the HuggingFaceAI class

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open("./prompts/contract_gen_no_negotiate_prompt.txt", "r") as file:
    PROMPT_TEMPLATE = file.read()

with open("freelanceer_contract_boilerplate.txt", "r") as boilerplate_file:
    BOILERPLATE_TEXT = boilerplate_file.read()

class ContractNegotiatorAI:
    def __init__(self):
        """
        Initializes the ContractNegotiatorAI.
        """
        self.huggingface_ai = HuggingFaceAI()  # Initialize HuggingFaceAI
        self.negotiation_pipeline = None
        self.tokenizer = None

    def setup_pipeline(self, task="text-generation", model="gpt2"):
        """
        Sets up the HuggingFace pipeline for text generation using HuggingFaceAI.
        """
        try:
            self.huggingface_ai.check_installed_software()  # Check required software
            self.negotiation_pipeline = self.huggingface_ai.setup_pipeline(task=task, model=model)
            if self.negotiation_pipeline:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model)  # Load tokenizer
                logging.info(f"HuggingFace pipeline initialized successfully with model: {model}.")
            else:
                logging.error("Failed to initialize HuggingFace pipeline.")
        except Exception as e:
            logging.error(f"Error during pipeline setup: {e}")
            self.negotiation_pipeline = None
            self.tokenizer = None

    def select_model(self, task="text-generation", model="gpt2"):
        """
        Dynamically selects the model and task for the pipeline.

        Args:
            task (str): The task to perform (e.g., "text-generation", "text2text-generation").
            model (str): The model to use (e.g., "gpt2", "google/flan-t5-large").
        """
        try:
            self.setup_pipeline(task=task, model=model)
            logging.info(f"Model selected: {model} for task: {task}.")
        except Exception as e:
            logging.error(f"Error selecting model: {e}")

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

    def generate_contract(self, freelancer_data, hirer_data, model_name="default_model"):
        """
        Generates a contract based on the provided freelancer and hirer data using the selected model.
        """
        if not self.negotiation_pipeline:
            logging.error("Pipeline not initialized. Please call select_model() first.")
            return None

        # Debugging: Log the PROMPT_TEMPLATE and data being passed
        logging.debug(f"PROMPT_TEMPLATE: {PROMPT_TEMPLATE}")
        logging.debug(f"Freelancer Data: {freelancer_data}")
        logging.debug(f"Hirer Data: {hirer_data}")

        # Clean and validate the PROMPT_TEMPLATE
        prompt_template_cleaned = PROMPT_TEMPLATE.strip()  # Remove unintended leading/trailing whitespace

        # Insert freelancer_name separately
        freelancer_name = freelancer_data.get('freelancer_name', 'Freelancer')
        prompt = prompt_template_cleaned.replace("{freelancer_name}", freelancer_name)

        # Use string concatenation for the rest of the prompt
        prompt = prompt.replace("{freelancer_title}", freelancer_data.get('freelancer_title', 'Professional'))
        prompt = prompt.replace("{job_title}", hirer_data.get('job_description', {}).get('title', 'Unknown Job Title'))
        prompt = prompt.replace("{job_description}", hirer_data.get('job_description', {}).get('description', 'No description provided.'))
        prompt = prompt.replace("{required_skills}", ", ".join(hirer_data.get('job_description', {}).get('required_skills', [])))
        prompt = prompt.replace("{estimated_duration_weeks}", str(hirer_data.get('job_description', {}).get('estimated_duration_weeks', 'Unknown')))
        prompt = prompt.replace("{freelancer_hourly_rate}", str(freelancer_data.get('expected_terms', {}).get('hourly_rate', 'Not specified')))
        prompt = prompt.replace("{freelancer_expected_hours_per_week}", str(freelancer_data.get('expected_terms', {}).get('expected_hours_per_week', 'Not specified')))
        prompt = prompt.replace("{freelancer_additional_expenses}", ", ".join(
            f"{key}: {value}" for key, value in freelancer_data.get('expected_terms', {}).get('additional_expenses', {}).items()
        ) if freelancer_data.get('expected_terms', {}).get('additional_expenses') else "None")
        prompt = prompt.replace("{hirer_hourly_rate}", str(hirer_data.get('expected_terms', {}).get('hourly_rate', 'Not specified')))
        prompt = prompt.replace("{hirer_expected_hours_per_week}", str(hirer_data.get('expected_terms', {}).get('expected_hours_per_week', 'Not specified')))
        prompt = prompt.replace("{hirer_budget}", str(hirer_data.get('expected_terms', {}).get('budget', 'Not specified')))
        prompt = prompt.replace("{agreed_hourly_rate}", "TBD")
        prompt = prompt.replace("{agreed_hours_per_week}", "TBD")
        prompt = prompt.replace("{agreed_expenses}", "TBD")

        logging.debug(f"Formatted Prompt: {prompt}")

        # Truncate the prompt to avoid exceeding the model's maximum sequence length
        truncated_prompt = self.truncate_prompt(prompt)

        # Generate text using the selected model
        generated_text = self.negotiation_pipeline(truncated_prompt, max_new_tokens=800, num_return_sequences=1)[0]['generated_text']

        # Save the generated contract to a text file
        output_dir = "generated_contracts"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"contract_{model_name}.txt")
        with open(output_file_path, "w") as f:
            f.write(generated_text)

        return generated_text

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
    try:
        with open("freelancer_data.json", "r") as freelancer_file:
            freelancer_data = json.load(freelancer_file)
    except Exception as e:
        print(f"Error loading freelancer data: {e}")
        freelancer_data = None

    try:
        with open("hirer_data.json", "r") as hirer_file:
            hirer_data = json.load(hirer_file)
    except Exception as e:
        print(f"Error loading hirer data: {e}")
        hirer_data = None

    negotiator = ContractNegotiatorAI()

    try:
        negotiator.setup_pipeline()  # Initialize the pipeline
    except Exception as e:
        print(f"Error setting up pipeline: {e}")

    # Test default model
    # try:
    contract = negotiator.generate_contract(freelancer_data, hirer_data)
    if contract:
        print("\nGenerated Contract:")
        print(contract)
    else:
        print("Failed to generate contract.")

    # Test text2text-generation model
    negotiator.select_model(task="text2text-generation", model="google/flan-t5-large")
    contract_text2text = negotiator.generate_contract(freelancer_data, hirer_data, model_name="google_flan_t5_large")
    if contract_text2text:
        print("\nGenerated Contract using text2text-generation model:")
        print(contract_text2text)
    else:
        print("Failed to generate contract using text2text-generation model.")

    # Test deepseek-ai/DeepSeek-V3 model
    negotiator.select_model(task="text-generation", model="deepseek-ai/DeepSeek-V3")
    contract_deepseek = negotiator.generate_contract(freelancer_data, hirer_data, model_name="deepseek_ai_DeepSeek_V3")
    if contract_deepseek:
        print("\nGenerated Contract using deepseek-ai/DeepSeek-V3 model:")
        print(contract_deepseek)
    else:
        print("Failed to generate contract using deepseek-ai/DeepSeek-V3 model.")

if __name__ == "__main__":
    test_negotiation()
