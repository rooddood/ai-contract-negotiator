# ai-contract-negotiator

Contributors can find more information on the current status of this project [here.](https://docs.google.com/document/d/1UkquQaXhuip0hjgGxJyqhRSuuH1hlVKdmIUohTOCW4o/edit?tab=t.0#heading=h.6jidvsd9f6nt)

# Setup
## Need Python 3.12.0
pyenv install -v 3.12.0

## Optionally set it as the global pyhton
pyenv global 3.12.0

## Create env and activate
python3.12 -m venv venv
source venv/bin/activate
pip install -r huggingface_test_requirements.txt 


# Running Tool
## Test of huggingface model alone
python HuggingFaceAI.py

## Contract negotiator implementation
python ContractNegotiatorAI.py


# Need to set api key to use external apis:
export GOOGLE_API_KEY='YOUR_GOOGLE_API_KEY'  # Replace with your key
# or
export OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
# or
export ANTHROPIC_API_KEY='YOUR_ANTHROPIC_API_KEY'
