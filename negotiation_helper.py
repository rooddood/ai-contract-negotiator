# import json

# def load_data(file_path):
#     """Loads data from a JSON file."""
#     with open(file_path, 'r') as f:
#         return json.load(f)

# def negotiate(freelancer_data, hirer_data):
#     """
#     Placeholder function for the negotiation logic using an LLM.
#     Currently, it just prints the input data.
#     """
#     print("Freelancer Data:")
#     print(json.dumps(freelancer_data, indent=4))
#     print("\nHirer Data:")
#     print(json.dumps(hirer_data, indent=4))

#     # In a real implementation, you would pass this data to an LLM
#     # and process its response to suggest negotiation points.
#     print("\n[LLM interaction would happen here to suggest negotiation points...]")

# if __name__ == "__main__":
#     # Create dummy JSON files for freelancer and hirer data
#     freelancer_info = {
#         "resume": {
#             "experience": [
#                 {"title": "Software Engineer", "company": "Tech Inc.", "years": "2020-2023", "skills": ["Python", "LLMs", "APIs"]},
#                 {"title": "Data Scientist", "company": "Data Corp.", "years": "2023-Present", "skills": ["Machine Learning", "Data Analysis", "Cloud Computing"]}
#             ],
#             "skills": ["Communication", "Problem-solving", "Negotiation"]
#         },
#         "expected_terms": {
#             "hourly_rate": 75,
#             "expected_hours_per_week": 20,
#             "additional_expenses": {"software_licenses": 50, "travel": 0}
#         }
#     }

#     hirer_info = {
#         "job_description": {
#             "title": "LLM Integration Specialist",
#             "description": "Seeking a freelancer to integrate and fine-tune LLMs for a new application.",
#             "required_skills": ["Python", "LLMs", "API Development", "Cloud Platforms"],
#             "estimated_duration_weeks": 12
#         },
#         "expected_terms": {
#             "hourly_rate": 60,
#             "expected_hours_per_week": 15,
#             "budget": 12000
#         }
#     }

#     # Save the data to JSON files
#     with open("freelancer_data.json", "w") as f:
#         json.dump(freelancer_info, f, indent=4)

#     with open("hirer_data.json", "w") as f:
#         json.dump(hirer_info, f, indent=4)

#     # Load the data from the JSON files
#     freelancer_data = load_data("freelancer_data.json")
#     hirer_data = load_data("hirer_data.json")

#     # Initiate the negotiation process
#     negotiate(freelancer_data, hirer_data)







import google.generativeai as genai
import json

# IMPORTANT: Replace with your actual Gemini API key
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def load_data(file_path):
    """Loads data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def negotiate_with_gemini(freelancer_data, hirer_data):
    """
    Interacts with the Gemini LLM to suggest negotiation points.
    """
    model = genai.GenerativeModel('gemini-pro')

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

    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    # Load data from JSON files (assuming these files exist from our previous interaction)
    try:
        freelancer_data = load_data("freelancer_data.json")
        hirer_data = load_data("hirer_data.json")

        # Initiate the negotiation process with Gemini
        negotiation_suggestions = negotiate_with_gemini(freelancer_data, hirer_data)
        print("\nNegotiation Suggestions from Gemini:")
        print(negotiation_suggestions)

    except FileNotFoundError:
        print("Error: Make sure 'freelancer_data.json' and 'hirer_data.json' exist in the same directory.")
        print("Please run the previous code block first to create these files.")