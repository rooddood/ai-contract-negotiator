import os
import csv
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

    def load_job_types(self, csv_path):
        job_types = {}
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                category = row['category'].strip()
                desc = row['job_type_description'].strip()
                job_types.setdefault(category, []).append(desc)
        return job_types

    def load_user_personas(self, csv_path):
        personas = []
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                persona = row['persona'].strip()
                focus = row['focus'].strip()
                personas.append(f"{persona}: {focus}")
        return personas

    def format_job_types_section(self, job_types):
        lines = []
        for category, jobs in job_types.items():
            lines.append(f"* **{category}:**")
            for job in jobs:
                lines.append(f"  * {job}")
        return '\n'.join(lines)

    def format_user_personas_section(self, personas):
        return '\n'.join([f"* {p}" for p in personas])

    def load_prompt(self, file_path: str, job_types=None, personas=None):
        """Loads the SOW generation prompt from a text file and injects job types and personas."""
        with open(file_path, 'r') as f:
            prompt = f.read()
        if job_types and personas:
            # Replace the Job Types and User Personas sections
            import re
            prompt = re.sub(r"\*\*Job Types:\*\*[\s\S]*?\*\*User Personas:\*\*", f"**Job Types:**\n{self.format_job_types_section(job_types)}\n\n**User Personas:**", prompt)
            prompt = re.sub(r"\*\*User Personas:\*\*[\s\S]*?Generate \*\*one\*\* JSON object", f"**User Personas:**\n{self.format_user_personas_section(personas)}\n\nGenerate **one** JSON object", prompt)
        
        return prompt

    def generate_sow(self, prompt_path=None):
        """Generates a Statement of Work (SOW) using an AI agent."""
        if prompt_path is None:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "sow_generation_prompt.txt")
        job_types_csv = os.path.join(os.path.dirname(__file__), "prompts", "sow_job_types.csv")
        personas_csv = os.path.join(os.path.dirname(__file__), "prompts", "sow_user_personas.csv")
        job_types = self.load_job_types(job_types_csv)
        personas = self.load_user_personas(personas_csv)
        prompt = self.load_prompt(prompt_path, job_types, personas)
        if self.provider == "openai":
            response = self.agent.get_response(
                prompt=prompt,
                model_name=self.model_name or "gpt-4.1"
            )
            try:
                sow_json = response.json()
            except Exception:
                sow_json = response.text if hasattr(response, 'text') else str(response)
        elif self.provider == "huggingface":
            # Tokenize and truncate prompt if needed
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)["input_ids"][0]
            prompt_token_count = prompt_tokens.shape[0]
            # Ensure total tokens (prompt + generated) do not exceed model max
            max_new_tokens = min(512, self.max_length - prompt_token_count)
            if max_new_tokens <= 0:
                # Truncate prompt further to allow at least 1 token generation
                prompt_tokens = prompt_tokens[-(self.max_length - 1):]
                prompt_token_count = prompt_tokens.shape[0]
                max_new_tokens = 1
            inputs = {"input_ids": prompt_tokens.unsqueeze(0)}
            if getattr(self, "hf_model_type", None) == "seq2seq":
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:  # causal LM (e.g., GPT-2)
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate_sow_for_each_jobtype_and_persona(self, prompt_path=None, output_dir="generated_sows"):
        """Generates a SOW for each combination of job type and persona, saves each as a JSON file."""
        import re
        import json
        if prompt_path is None:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "sow_generation_prompt.txt")
        job_types_csv = os.path.join(os.path.dirname(__file__), "prompts", "sow_job_types.csv")
        personas_csv = os.path.join(os.path.dirname(__file__), "prompts", "sow_user_personas.csv")
        # Load job types and personas as flat lists
        job_types = []
        with open(job_types_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                job_types.append((row['category'].strip(), row['job_type_description'].strip()))
        personas = []
        with open(personas_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                personas.append((row['persona'].strip(), row['focus'].strip()))
        # Load the prompt template
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()
        # Regex to find persona and job type lines
        persona_pattern = re.compile(r'\* "([^"]+)": ([^\n]+)')
        persona_match = persona_pattern.search(prompt_template)
        if not persona_match:
            raise ValueError("Default persona not found in prompt template.")
        default_persona_line = persona_match.group(0)
        jobtype_pattern = re.compile(r'  \* (.+)')
        jobtype_match = jobtype_pattern.search(prompt_template)
        default_jobtype_line = jobtype_match.group(0) if jobtype_match else None

        # Ensure output directory exists
        # os.makedirs(output_dir, exist_ok=True)
        
        for category, job_type in job_types:
            for persona_name, persona_focus in personas:
                # Build job types and personas sections for this combination
                job_types_section = f"* **{category}:**\n  * {job_type}"
                personas_section = f"* {persona_name}: {persona_focus}"
                # Replace the Job Types and User Personas sections
                prompt = re.sub(r"\*\*Job Types:\*\*[\s\S]*?\*\*User Personas:\*\*", f"**Job Types:**\n{job_types_section}\n\n**User Personas:**", prompt_template)
                prompt = re.sub(r"\*\*User Personas:\*\*[\s\S]*?Generate \*\*one\*\* JSON object", f"**User Personas:**\n{personas_section}\n\nGenerate **one** JSON object", prompt)
                # Replace the default persona line with the current persona
                persona_line = f'* "{persona_name}": {persona_focus}'
                prompt = prompt.replace(default_persona_line, persona_line)
                # Replace the default job type line with the current job type (if present)
                if default_jobtype_line:
                    prompt = prompt.replace(default_jobtype_line, f"  * {job_type}")
                # Progress output
                print(f"Generating SOW for job type: {job_type} | persona: {persona_name}")
                # Get response
                if self.provider == "openai":
                    response = self.agent.get_response(
                        prompt=prompt,
                        model_name=self.model_name or "gpt-4.1"
                    )
                    sow_text = response.text if hasattr(response, 'text') else str(response)
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
                        sow_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    else:
                        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
                        sow_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                # Try to parse as JSON
                print(f"[DEBUG] Raw SOW text length: {len(sow_text)}")
                print(f"[DEBUG] Raw SOW text (first 500 chars):\n{sow_text[:500]}")
                try:
                    sow_json = json.loads(sow_text)
                except Exception as e:
                    print(f"ERROR: Output is not valid JSON for job type '{job_type}' and persona '{persona_name}'.")
                    print(f"[DEBUG] Full SOW text:\n{sow_text}")
                    raise e
                # Safe filename
                def safe(s):
                    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)[:40]
                filename = f"sow_{safe(category)}_{safe(job_type)}_{safe(persona_name)}.json"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w') as f:
                    json.dump(sow_json, f, indent=2)
                print(f"Saved: {filepath}")

if __name__ == "__main__":
    generator = SOWGenerator(provider="huggingface", model_name="gpt2")
    generator.generate_sow_for_each_jobtype_and_persona()
