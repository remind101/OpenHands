from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_get_pr_diff_success(MockAsyncClient, github_handler):
    """Tests successful retrieval of PR diff using AsyncClient."""
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.text = 'sample diff content'
    mock_response.headers = {'content-type': 'application/vnd.github.v3.diff'}
    # mock_response.raise_for_status = MagicMock() # Removed - rely on status_code

    # Configure the mock client instance returned by the context manager
    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
    mock_client_instance.get.return_value = mock_response

    # Call the async method
    diff = await github_handler.get_pr_diff(PR_NUM)

    assert diff == 'sample diff content'
    expected_headers = {
        'Accept': 'application/vnd.github.v3.diff',
        'Authorization': 'token test-token',  # Use real token for assertion
        'X-GitHub-Api-Version': '2022-11-28',
    }
    expected_url = f'https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUM}'

    # Check that the get method was called correctly on the client instance
    mock_response.raise_for_status.assert_called_once()  # Verify raise_for_status was called
    mock_client_instance.get.assert_called_once_with(
        expected_url, headers=expected_headers
    )
    mock_response.raise_for_status.assert_called_once()  # Verify raise_for_status was called


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_get_pr_diff_error(MockAsyncClient, github_handler):
    """Tests error handling when fetching PR diff fails using AsyncClient."""
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.text = 'Not Found'
    mock_http_error = httpx.HTTPStatusError(
        'Not Found',
        request=MagicMock(),
        response=mock_response,
    )
    # raise_for_status is synchronous
    mock_response.raise_for_status = MagicMock(side_effect=mock_http_error)

    # Configure the mock client instance
    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
    mock_client_instance.get.return_value = mock_response

    expected_url = f'https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUM}'
    expected_headers = {
        'Accept': 'application/vnd.github.v3.diff',
        'Authorization': 'token test-token',  # Use real token for assertion
        'X-GitHub-Api-Version': '2022-11-28',
    }

    with pytest.raises(httpx.HTTPStatusError):
        await github_handler.get_pr_diff(PR_NUM)

    # Assertions *after* the expected exception
    mock_client_instance.get.assert_called_once_with(
        expected_url, headers=expected_headers
    )
    mock_response.raise_for_status.assert_called_once()  # Verify raise_for_status was called


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_post_review_single_comment(MockAsyncClient, github_handler):
    """Tests posting a review with a single comment using AsyncClient."""
    mock_post_response = AsyncMock(spec=httpx.Response)
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {'id': 1}
    mock_post_response.raise_for_status = MagicMock()

    # Configure the mock client instance
    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
    mock_client_instance.post.return_value = mock_post_response

    comments = [ReviewComment(path='file1.py', line=10, comment='First comment')]
    await github_handler.post_review(pr_number=PR_NUM, comments=comments)

    expected_url = f'https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUM}/reviews'
    expected_headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': 'token **********',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    # Updated expected body based on refactored post_review logic
    expected_payload = {
        'body': 'OpenHands AI Code Review:\n\n**Line-Specific Feedback:** (see comments below)',
        'event': 'COMMENT',
        'comments': [{'path': 'file1.py', 'line': 10, 'body': 'First comment'}],
    }

    mock_client_instance.post.assert_called_once()
    args, kwargs = mock_client_instance.post.call_args
    assert args[0] == expected_url
    assert kwargs['headers'] == expected_headers
    assert kwargs['json'] == expected_payload
    mock_post_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_post_review_multiple_comments(MockAsyncClient, github_handler):
    """Tests posting a review with multiple comments using AsyncClient."""
    mock_post_response = AsyncMock(spec=httpx.Response)
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {'id': 2}
    mock_post_response.raise_for_status = MagicMock()

    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
    mock_client_instance.post.return_value = mock_post_response

    comments = [
        ReviewComment(path='file1.py', line=10, comment='First comment'),
        ReviewComment(path='file2.py', line=25, comment='Second comment'),
    ]
    await github_handler.post_review(pr_number=PR_NUM, comments=comments)

    expected_url = f'https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUM}/reviews'
    expected_headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': 'token **********',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    # Updated expected body based on refactored post_review logic
    expected_payload = {
        'body': 'OpenHands AI Code Review:\n\n**Line-Specific Feedback:** (see comments below)',
        'event': 'COMMENT',
        'comments': [
            {'path': 'file1.py', 'line': 10, 'body': 'First comment'},
            {'path': 'file2.py', 'line': 25, 'body': 'Second comment'},
        ],
    }

    mock_client_instance.post.assert_called_once()
    args, kwargs = mock_client_instance.post.call_args
    assert args[0] == expected_url
    assert kwargs['headers'] == expected_headers
    assert kwargs['json'] == expected_payload
    mock_post_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_post_review_no_comments(MockAsyncClient, github_handler):
    """Tests posting a review with no comments using AsyncClient."""
    mock_post_response = AsyncMock(spec=httpx.Response)
    mock_post_response.status_code = 200
    mock_post_response.raise_for_status = MagicMock()

    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
    mock_client_instance.post.return_value = mock_post_response

    await github_handler.post_review(pr_number=PR_NUM, comments=[])

    expected_url = f'https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUM}/reviews'
    expected_headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': 'token **********',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    # Updated expected body based on refactored post_review logic
    expected_payload = {
        'body': 'OpenHands AI Code Review:',
        'event': 'COMMENT',
        'comments': [],
    }

    mock_client_instance.post.assert_called_once()
    args, kwargs = mock_client_instance.post.call_args
    assert args[0] == expected_url
    assert kwargs['headers'] == expected_headers
    assert kwargs['json'] == expected_payload
    mock_post_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_post_review_api_error(MockAsyncClient, github_handler):
    """Tests error handling when posting review fails using AsyncClient."""
    mock_post_response = AsyncMock(spec=httpx.Response)
    mock_post_response.status_code = 400
    mock_post_response.request = MagicMock(url='dummy_url')
    mock_post_response.json.return_value = {'message': 'Validation Failed'}
    mock_http_error = httpx.HTTPStatusError(
        'API Error', request=mock_post_response.request, response=mock_post_response
    )
    mock_post_response.raise_for_status = MagicMock(side_effect=mock_http_error)

    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
    mock_client_instance.post.return_value = mock_post_response

    comments = [ReviewComment(path='file1.py', line=10, comment='Error comment')]
    with pytest.raises(httpx.HTTPStatusError):
        await github_handler.post_review(pr_number=PR_NUM, comments=comments)

    # Verify post was called
    mock_client_instance.post.assert_called_once()
    # Verify raise_for_status was called on the response mock
    mock_post_response.raise_for_status.assert_called_once()
