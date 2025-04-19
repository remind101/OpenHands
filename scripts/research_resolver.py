import os
import sys
import json
import requests

# Constants
GITHUB_API_URL = "https://api.github.com"
PROCESSED_LABEL = "research-answered" # Label to add after processing

def get_env_var(name):
    """Gets an environment variable or exits if not found."""
    value = os.getenv(name)
    if not value:
        print(f"Error: Environment variable {name} not set.")
        sys.exit(1)
    return value

def make_github_request(method, url, token, json_data=None):
    """Makes a request to the GitHub API."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=json_data)
        elif method.upper() == 'PATCH':
             response = requests.patch(url, headers=headers, json=json_data)
        else:
            print(f"Unsupported HTTP method: {method}")
            sys.exit(1)

        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        # Check rate limit
        if 'X-RateLimit-Remaining' in response.headers:
            print(f"Rate limit remaining: {response.headers['X-RateLimit-Remaining']}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"GitHub API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Failed to decode JSON response from GitHub API.")
        print(f"Response text: {response.text}")
        sys.exit(1)


def check_if_processed(issue_data):
    """Checks if the issue has the PROCESSED_LABEL."""
    labels = issue_data.get("labels", [])
    for label in labels:
        if label.get("name") == PROCESSED_LABEL:
            print(f"Issue #{issue_data['number']} already has the '{PROCESSED_LABEL}' label. Skipping.")
            return True
    # TODO: Could also check for a specific marker comment if preferred
    return False

def call_llm(prompt, api_key):
    """Placeholder function to call the LLM API."""
    # Replace this with the actual API call to your chosen LLM
    # Example: Using OpenAI's API (requires `openai` package)
    # from openai import OpenAI
    # client = OpenAI(api_key=api_key)
    # try:
    #     response = client.chat.completions.create(
    #         model="gpt-4-turbo", # Or your preferred model
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant analyzing GitHub issues."},
    #             {"role": "user", "content": prompt}
    #         ]
    #     )
    #     return response.choices[0].message.content
    # except Exception as e:
    #     print(f"Error calling LLM: {e}")
    #     return "Error: Could not get analysis from LLM."

    print("--- LLM Call (Simulated) ---")
    print(f"Prompt:
{prompt}")
    print("-----------------------------")
    # Simulate a response for now
    return f"This is a simulated analysis based on the issue content. The LLM would provide a more detailed answer here regarding effort estimation, breakdown, or feasibility based on:

{prompt}"

def main():
    print("Starting research resolver script...")

    # Get required environment variables
    github_token = get_env_var("GITHUB_TOKEN")
    llm_api_key = get_env_var("LLM_API_KEY") # Ensure this secret is set in GitHub repo settings
    issue_number = get_env_var("ISSUE_NUMBER")
    repo_context_json = get_env_var("REPO_CONTEXT")
    issue_context_json = get_env_var("ISSUE_CONTEXT")

    try:
        repo_context = json.loads(repo_context_json)
        issue_context = json.loads(issue_context_json)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON context: {e}")
        sys.exit(1)

    repo_full_name = repo_context.get("full_name")
    if not repo_full_name:
        print("Error: Could not determine repository full name from context.")
        sys.exit(1)

    # 1. Fetch full issue details (issue_context might be partial)
    issue_url = f"{GITHUB_API_URL}/repos/{repo_full_name}/issues/{issue_number}"
    print(f"Fetching issue details from: {issue_url}")
    issue_data = make_github_request("GET", issue_url, github_token)
    print(f"Successfully fetched details for issue #{issue_number}")

    # 2. Check if already processed
    if check_if_processed(issue_data):
        sys.exit(0) # Exit successfully, nothing more to do

    # 3. Prepare LLM Prompt
    issue_title = issue_data.get("title", "")
    issue_body = issue_data.get("body", "") or "No description provided." # Handle empty body

    # Basic prompt - can be significantly enhanced
    prompt = f"""
Analyze the following GitHub issue from the repository '{repo_full_name}'.
The issue asks for research or analysis. Please provide an answer addressing the core question(s).

Consider the following aspects based on the issue content:
- Feasibility: Is the request possible?
- Existing Work: Does any part of this feature/idea already exist in the codebase (conceptually)?
- Effort Estimation: Provide a rough estimate of the effort required (e.g., Small, Medium, Large).
- Breakdown: If applicable, break the goal down into smaller, implementable steps.

Issue Title: {issue_title}

Issue Body:
{issue_body}

---
Provide your analysis below:
"""

    # (Optional Enhancement: Add code context retrieval here using git/grep if needed)

    # 4. Call LLM
    print("Calling LLM for analysis...")
    llm_response = call_llm(prompt, llm_api_key)
    print("Received analysis from LLM.")

    # 5. Post Comment to GitHub Issue
    comment_url = f"{GITHUB_API_URL}/repos/{repo_full_name}/issues/{issue_number}/comments"
    comment_body = f"Automated analysis for research issue:

{llm_response}

---
*This comment was generated by the Research Resolver action.*"
    print(f"Posting comment to issue #{issue_number}...")
    make_github_request("POST", comment_url, github_token, json_data={"body": comment_body})
    print("Successfully posted comment.")

    # 6. Add Processed Label to GitHub Issue
    labels_url = f"{GITHUB_API_URL}/repos/{repo_full_name}/issues/{issue_number}/labels"
    print(f"Adding label '{PROCESSED_LABEL}' to issue #{issue_number}...")
    make_github_request("POST", labels_url, github_token, json_data={"labels": [PROCESSED_LABEL]})
    print(f"Successfully added label '{PROCESSED_LABEL}'.")

    print("Research resolver script finished successfully.")

if __name__ == "__main__":
    main()
