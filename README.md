# ai-contract-negotiator

python -m venv venv
pip install requests
source venv/bin/activate

#Test of huggingface model
python huggingface_test.py

#contract negotiator implementation
python negotiation_helper.py


# Need to set api key:
export GOOGLE_API_KEY='YOUR_GOOGLE_API_KEY'  # Replace with your key
# or
export OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
# or
export ANTHROPIC_API_KEY='YOUR_ANTHROPIC_API_KEY'