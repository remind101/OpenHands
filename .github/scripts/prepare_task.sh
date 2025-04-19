#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

TASK_FILE="/tmp/agent_task.txt"

# Check if GITHUB_EVENT_PATH is set and the file exists
if [ -z "$GITHUB_EVENT_PATH" ] || [ ! -f "$GITHUB_EVENT_PATH" ]; then
  echo "Error: GITHUB_EVENT_PATH is not set or file does not exist." >&2
  exit 1
fi

# Use jq to handle potential quotes and newlines in title/body
ISSUE_TITLE=$(jq -r '.issue.title' "$GITHUB_EVENT_PATH")
ISSUE_BODY=$(jq -r '.issue.body // ""' "$GITHUB_EVENT_PATH") # Use default empty string if body is null
ISSUE_NUMBER=$(jq -r '.issue.number' "$GITHUB_EVENT_PATH")
ISSUE_URL=$(jq -r '.issue.html_url' "$GITHUB_EVENT_PATH")
REPO_NAME=$(jq -r '.repository.full_name' "$GITHUB_EVENT_PATH")

# Create the task file content using cat << EOF
# Note: Indentation within the heredoc doesn't affect the output file,
# but helps visually distinguish it from the script logic.
cat << EOF > "$TASK_FILE"
Task for Research Agent:
=========================
Issue Number: $ISSUE_NUMBER
Issue URL: $ISSUE_URL
Repository: $REPO_NAME
-------------------------
Issue Title: $ISSUE_TITLE
-------------------------
Issue Body:
$ISSUE_BODY
-------------------------
Request:
Please analyze this issue. Your goal is to provide the research requested.
Use your available tools (web browsing, code analysis, file reading/writing, command execution) to gather information.
Focus on answering the core questions, which might involve:
- Assessing feasibility.
- Identifying existing relevant code or features.
- Estimating the effort required (e.g., Small, Medium, Large).
- Breaking down the goal into implementable steps.
Provide a clear, concise summary of your findings as your final output.
Make sure your final response is suitable for posting as a GitHub comment.
EOF

echo "Task file created at $TASK_FILE"
# Print task file content to logs for debugging
echo "--- Task File Content Start ---"
cat "$TASK_FILE"
echo "--- Task File Content End ---"

# Output the task file path for the workflow
# Using the specific format ::set-output name=task_file::/path/to/file
echo "::set-output name=task_file::$TASK_FILE"
