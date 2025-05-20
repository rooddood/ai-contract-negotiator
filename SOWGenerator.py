import os
import csv
import pandas as pd  # Use pandas for CSV loading
from AIAgent import AIAgent, SOW  # Import SOW class

class SOWGenerator:
    def __init__(self, provider="openai", api_key=None, model_name=None):
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        if self.provider == "openai":
            self.agent = AIAgent(provider=self.provider, api_key=self.api_key)
        elif self.provider == "huggingface":
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
                import torch
                self.model_name = self.model_name or "google/flan-t5-large"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                config = AutoConfig.from_pretrained(self.model_name)
                self.max_length = getattr(config, "max_position_embeddings", 1024)
                if config.is_encoder_decoder:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                    self.hf_model_type = "seq2seq"
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    self.hf_model_type = "causal"
            except ImportError:
                raise ImportError("transformers library is required for HuggingFace provider.")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Load job types and personas as DataFrames (handle commas in fields)
        base_dir = os.path.dirname(__file__)
        self.job_types_df = pd.read_csv(os.path.join(base_dir, "prompts", "sow_job_types.csv"), quotechar='"', skipinitialspace=True)
        self.personas_df = pd.read_csv(
            os.path.join(base_dir, "prompts", "sow_user_personas.csv"),
            quotechar='"', skipinitialspace=True, on_bad_lines='skip'
        )
        # Load the prompt template once
        prompt_path = os.path.join(base_dir, "prompts", "sow_generation_prompt.txt")
        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()

    def get_job_types_dict(self):
        """Return job types as a dict: {category: [(job_type, description), ...]}"""
        job_types = {}
        for _, row in self.job_types_df.iterrows():
            category = row['category'].strip()
            job_type = row['job_type'].strip()
            description = row['description'].strip()
            job_types.setdefault(category, []).append((job_type, description))
        return job_types

    def get_personas_list(self):
        """Return personas as a list of 'persona: focus' strings, skipping NaN values."""
        personas = []
        for _, row in self.personas_df.iterrows():
            persona = row['persona'] if pd.notnull(row['persona']) else ''
            focus = row['focus'] if pd.notnull(row['focus']) else ''
            personas.append(f"{str(persona).strip()}: {str(focus).strip()}")
        return [p for p in personas if p != ':']

    def get_all_job_types_str(self):
        """Return a formatted string of all job types grouped by category."""
        job_types = self.get_job_types_dict()
        lines = []
        for category, jobtype_tuples in job_types.items():
            #TODO: Maybe comment this?
            lines.append(f"{category}:")
            for job_type, description in jobtype_tuples:
                lines.append(f"  - {job_type}: {description}")
        return "\n".join(lines)

    def get_all_personas_str(self):
        """Return a formatted string of all personas with their focus."""
        personas = []
        for _, row in self.personas_df.iterrows():
            persona = row['persona'] if pd.notnull(row['persona']) else ''
            focus = row['focus'] if pd.notnull(row['focus']) else ''
            if persona and focus:
                personas.append(f"- {persona}: {focus}")
        return "\n".join(personas)

    def generate_sow(self, prompt):
        """Generates a Statement of Work (SOW) using an AI agent for a single job type and persona, given a prompt string."""
        print("[INFO] Starting SOW generation...")
        print(f"[INFO] Final prompt sent to model:\n{prompt}")
        # ...existing code for model call and saving output...
        if self.provider == "openai":
            response = self.agent.get_response(
                prompt=prompt,
                model_name=self.model_name or "gpt-4.1"
            )
            try:
                sow_json = response.json()
            except Exception:
                sow_json = response.text if hasattr(response, 'text') else str(response)
            print("[INFO] SOW generated using OpenAI provider.")
            print(f"[DEBUG] SOW output (first 500 chars):\n{str(sow_json)[:500]}")
            # Save SOW output to file
            output_dir = os.path.join(os.path.dirname(__file__), "generated_sows")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "sow_generated.json")
            with open(output_path, 'w') as f:
                import json
                json.dump(sow_json, f, indent=2) if isinstance(sow_json, dict) else f.write(str(sow_json))
            print(f"[INFO] SOW saved to: {output_path}")
        elif self.provider == "huggingface":
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)["input_ids"][0]
            prompt_token_count = prompt_tokens.shape[0]
            max_new_tokens = min(512, self.max_length - prompt_token_count)
            if max_new_tokens <= 0:
                prompt_tokens = prompt_tokens[-(self.max_length - 1):]
                prompt_token_count = prompt_tokens.shape[0]
                max_new_tokens = 1
            inputs = {"input_ids": prompt_tokens.unsqueeze(0)}
            if getattr(self, "hf_model_type", None) == "seq2seq":
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("[INFO] SOW generated using HuggingFace provider.")
            print(f"[DEBUG] SOW output (first 500 chars):\n{generated_text[:500]}")
            # Save SOW output to file
            output_dir = os.path.join(os.path.dirname(__file__), "generated_sows")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "sow_generated.json")
            with open(output_path, 'w') as f:
                import json
                try:
                    sow_json = json.loads(generated_text)
                    json.dump(sow_json, f, indent=2)
                except Exception:
                    f.write(generated_text)
            print(f"[INFO] SOW saved to: {output_path}")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        print("[INFO] SOW generation complete.")

    def get_category_for_job_type(self, job_type, description=None):
        """Return the category for a given job_type and (optionally) description."""
        if description is not None:
            row = self.job_types_df[(self.job_types_df['job_type'].str.strip() == job_type.strip()) & (self.job_types_df['description'].str.strip() == description.strip())]
            if not row.empty:
                return row.iloc[0]['category']
        else:
            row = self.job_types_df[self.job_types_df['job_type'].str.strip() == job_type.strip()]
            if not row.empty:
                return row.iloc[0]['category']
        return "Unknown"

    def generate_sow_for_job_and_persona(self, job_category, job_type, persona_name, persona_focus, description=None):
        """Generate a SOW for a specific job type and persona, including all job types and personas in the prompt, and category."""
        prompt = self.prompt_template
        # If description is provided, use it in the prompt
        if description:
            job_type_str = f"{job_type}: {description}"
        else:
            job_type_str = job_type
        prompt = prompt.replace("{job_type}", job_type_str)
        prompt = prompt.replace("{persona_name}", persona_name)
        prompt = prompt.replace("{persona_focus}", persona_focus)
        # Add category
        prompt = prompt.replace("{job_category}", job_category)
        self.generate_sow(prompt)

    def extract_json_from_text(self, text):
        import re, json
        match = re.search(r'({[\s\S]*?})', text)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        match = re.search(r'(\[[\s\S]*?\])', text)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        return None

if __name__ == "__main__":
    generator = SOWGenerator(provider="huggingface", model_name="google/flan-t5-large")
    # Example: Generate SOW for 'Software Development' and 'Efficient Manager'
    job_category = "Technology"
    job_type = "Software Development: Building a customer relationship management (CRM) system."
    persona_name = "Efficient Manager"
    persona_focus = "Focused on clear deliverables, timelines, and cost."
    generator.generate_sow_for_job_and_persona(job_category, job_type, persona_name, persona_focus)
    # generator.generate_sow_for_each_jobtype_and_persona()
