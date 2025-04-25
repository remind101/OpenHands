from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from openhands.code_reviewer.reviewer_output import ReviewComment
from openhands.resolver.interfaces.github import GithubPRHandler

OWNER = 'test-owner'
REPO = 'test-repo'
PR_NUM = 123
TOKEN = SecretStr('test-token')


@pytest.fixture
def github_handler():
    return GithubPRHandler(token=TOKEN, owner=OWNER, repo=REPO)


@patch('openhands.resolver.interfaces.github.httpx.get')
def test_get_pr_diff_success(mock_get, github_handler):
    """Tests successful retrieval of PR diff."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.text = 'sample diff content'
    mock_response.headers = {'content-type': 'application/vnd.github.v3.diff'}
    mock_response.raise_for_status = MagicMock()  # Mock raise_for_status for success
    mock_get.return_value = mock_response  # Configure the mock get function
    diff = github_handler.get_pr_diff(PR_NUM)

    assert diff == 'sample diff content'
    expected_headers = {
        'Accept': 'application/vnd.github.v3.diff',
        'Authorization': f'token {TOKEN.get_secret_value()}',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    expected_url = f'https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUM}'
    mock_get.assert_called_once_with(expected_url, headers=expected_headers)
    mock_response.raise_for_status.assert_called_once()  # Verify raise_for_status was called


@patch('openhands.resolver.interfaces.github.httpx.get')
def test_get_pr_diff_error(mock_get, github_handler):
    """Tests error handling when fetching PR diff fails."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.text = 'Not Found'
    # Configure the mock client and its get method to raise an error via raise_for_status
    mock_http_error = httpx.HTTPStatusError(
        'Not Found',
        request=MagicMock(),
        response=mock_response,  # Use mock_response here
    )
    mock_response.raise_for_status = MagicMock(side_effect=mock_http_error)
    expected_url = f'https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUM}'
    expected_headers = {
        'Accept': 'application/vnd.github.v3.diff',
        'Authorization': f'token {TOKEN.get_secret_value()}',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    mock_get.return_value = mock_response
    with pytest.raises(httpx.HTTPStatusError):
        github_handler.get_pr_diff(PR_NUM)
    mock_get.assert_called_once_with(expected_url, headers=expected_headers)
    mock_response.raise_for_status.assert_called_once()


@patch('openhands.resolver.interfaces.github.httpx.post')
async def test_post_review_single_comment(mock_post, github_handler):
    """Tests posting a review with a single comment."""
    mock_post_response = MagicMock(spec=httpx.Response)
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {'id': 1}  # Simulate successful review post
    # Simulate raise_for_status behavior for success response
    mock_post_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_post_response  # Configure the mock post function

    comments = [ReviewComment(path='file1.py', line=10, comment='First comment')]
    await github_handler.post_review(pr_number=PR_NUM, comments=comments)

    expected_url = f'https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUM}/reviews'
    expected_headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {TOKEN.get_secret_value()}',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    expected_payload = {
        'body': 'OpenHands AI Code Review:\n\n**Line-Specific Feedback:** (see comments below)',
        'event': 'COMMENT',
        'comments': [{'path': 'file1.py', 'line': 10, 'body': 'First comment'}],
    }
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == expected_url
    assert kwargs['headers'] == expected_headers
    assert kwargs['json'] == expected_payload


@pytest.mark.asyncio
@patch('openhands.resolver.interfaces.github.httpx.post')
async def test_post_review_multiple_comments(mock_post, github_handler):
    """Tests posting a review with multiple comments."""
    mock_post_response = MagicMock(spec=httpx.Response)
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {'id': 2}
    mock_post_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_post_response

    comments = [
        ReviewComment(path='file1.py', line=10, comment='First comment'),
        ReviewComment(path='file2.py', line=25, comment='Second comment'),
    ]
    await github_handler.post_review(pr_number=PR_NUM, comments=comments)

    expected_url = f'https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUM}/reviews'
    expected_headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {TOKEN.get_secret_value()}',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    expected_payload = {
        'body': 'OpenHands AI Code Review:\n\n**Line-Specific Feedback:** (see comments below)',
        'event': 'COMMENT',
        'comments': [
            {'path': 'file1.py', 'line': 10, 'body': 'First comment'},
            {'path': 'file2.py', 'line': 25, 'body': 'Second comment'},
        ],
    }
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == expected_url
    assert kwargs['headers'] == expected_headers
    assert kwargs['json'] == expected_payload
    mock_post_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
@patch('openhands.resolver.interfaces.github.httpx.post')
async def test_post_review_no_comments(mock_post, github_handler):
    """Tests posting a review with no comments (should not call API)."""
    await github_handler.post_review(pr_number=PR_NUM, comments=[])

    expected_url = f'https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUM}/reviews'
    expected_headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {TOKEN.get_secret_value()}',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    expected_payload = {
        'body': 'OpenHands AI Code Review:\n\n',
        'event': 'COMMENT',
        'comments': [],
    }
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == expected_url
    assert kwargs['headers'] == expected_headers
    assert kwargs['json'] == expected_payload
    # We also need to mock the response for this call
    mock_post_response = MagicMock(spec=httpx.Response)
    mock_post_response.status_code = 200
    mock_post_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_post_response
    mock_post_response.raise_for_status.assert_called_once()


@patch('openhands.resolver.interfaces.github.httpx.post')
async def test_post_review_api_error(mock_post, github_handler):
    """Tests error handling when posting review fails."""
    mock_post_response = MagicMock(spec=httpx.Response)
    mock_post_response.status_code = 400  # Simulate a client error
    mock_post_response.request = MagicMock(url='dummy_url')
    mock_post_response.json.return_value = {'message': 'Validation Failed'}

    mock_post_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        'API Error', request=mock_post_response.request, response=mock_post_response
    )
    mock_post.return_value = mock_post_response
    comments = [ReviewComment(path='file1.py', line=10, comment='Error comment')]
    with pytest.raises(httpx.HTTPStatusError):
        await github_handler.post_review(pr_number=PR_NUM, comments=comments)
