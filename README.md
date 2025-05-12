# ai-contract-negotiator

This tool is designed to explore the use of different models for automatic contract negotiation and generation. It leverages large language models (LLMs) from Hugging Face to facilitate these tasks.

Contributors can find more information on the current status of this project [here.](https://docs.google.com/document/d/1UkquQaXhuip0hjgGxJyqhRSuuH1hlVKdmIUohTOCW4o/edit?tab=t.0#heading=h.6jidvsd9f6nt)

# Setup
## Need Python 3.12.0
```bash
pyenv install -v 3.12.0
```

## Optionally set it as the global python
```bash
pyenv global 3.12.0
```

## Create env and activate
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r huggingface_test_requirements.txt
```

# Running Tool
## Test of huggingface model alone
```bash
python HuggingFaceAI.py
```

## Contract negotiator implementation
```bash
python ContractNegotiatorAI.py
```

