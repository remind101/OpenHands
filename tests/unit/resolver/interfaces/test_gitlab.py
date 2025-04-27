from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from openhands.code_reviewer.reviewer_output import ReviewComment
from openhands.resolver.interfaces.gitlab import GitlabPRHandler

OWNER = 'test-group/test-subgroup'  # GitLab uses group/subgroup/project
REPO = 'test-repo'
PR_NUM = 456  # Use a different number for clarity
TOKEN = SecretStr('test-gitlab-token')
BASE_DOMAIN = 'gitlab.example.com'
BASE_URL = f'https://{BASE_DOMAIN}/api/v4/projects/{OWNER.replace("/", "%2F")}%2F{REPO}'


@pytest.fixture
def gitlab_handler():
    # Note: GitLab owner/repo structure might differ, adjust fixture if needed
    return GitlabPRHandler(token=TOKEN, owner=OWNER, repo=REPO, base_domain=BASE_DOMAIN)


# ================================== get_pr_diff ==================================


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_gitlab_get_pr_diff_success(MockAsyncClient, gitlab_handler):
    """Tests successful retrieval of GitLab MR diff."""
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    # GitLab diff endpoint returns a list of diff versions
    mock_response.json.return_value = [{'diff': 'sample gitlab diff'}]
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
    mock_client_instance.get.return_value = mock_response

    diff = await gitlab_handler.get_pr_diff(PR_NUM)

    assert diff == 'sample gitlab diff'
    expected_url = f'{BASE_URL}/merge_requests/{PR_NUM}/diffs'
    expected_headers = {
        'Authorization': 'Bearer **********',  # Expect masked token
        'Accept': 'application/json',
    }
    mock_client_instance.get.assert_called_once_with(
        expected_url, headers=expected_headers
    )
    mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_gitlab_get_pr_diff_no_diff_found(MockAsyncClient, gitlab_handler):
    """Tests GitLab MR diff retrieval when the response is empty or lacks 'diff'."""
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = []  # Empty list
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
    mock_client_instance.get.return_value = mock_response

    diff = await gitlab_handler.get_pr_diff(PR_NUM)
    assert diff == ''  # Expect empty string

    # Test with missing 'diff' key
    mock_response.json.return_value = [{'no_diff_key': 'something'}]
    diff = await gitlab_handler.get_pr_diff(PR_NUM)
    assert diff == ''


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_gitlab_get_pr_diff_error(MockAsyncClient, gitlab_handler):
    """Tests error handling when fetching GitLab MR diff fails."""
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.text = 'Not Found'
    mock_http_error = httpx.HTTPStatusError(
        'Not Found', request=MagicMock(), response=mock_response
    )
    mock_response.raise_for_status = MagicMock(side_effect=mock_http_error)

    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
    mock_client_instance.get.return_value = mock_response

    with pytest.raises(httpx.HTTPStatusError):
        await gitlab_handler.get_pr_diff(PR_NUM)

    expected_url = f'{BASE_URL}/merge_requests/{PR_NUM}/diffs'
    mock_client_instance.get.assert_called_once_with(
        expected_url, headers=gitlab_handler.headers
    )
    mock_response.raise_for_status.assert_called_once()


# ================================== post_review ==================================


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_gitlab_post_review_success(MockAsyncClient, gitlab_handler):
    """Tests successful posting of a review comment to GitLab MR."""
    # Mock response for fetching MR details (needed for position)
    mock_get_response = AsyncMock(spec=httpx.Response)
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = {
        'iid': PR_NUM,
        'target_project_id': 12345,
        'diff_refs': {
            'base_sha': 'abc',
            'start_sha': 'def',
            'head_sha': 'ghi',
        },
    }
    mock_get_response.raise_for_status = MagicMock()

    # Mock response for posting the discussion
    mock_post_response = AsyncMock(spec=httpx.Response)
    mock_post_response.status_code = 201  # GitLab returns 201 Created
    mock_post_response.json.return_value = {'id': 'discussion_id'}
    mock_post_response.raise_for_status = MagicMock()

    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
    # Set up side effects for get (details) and post (comment)
    mock_client_instance.get.return_value = mock_get_response
    mock_client_instance.post.return_value = mock_post_response

    comments = [ReviewComment(path='src/main.py', line=50, comment='GitLab comment')]
    await gitlab_handler.post_review(pr_number=PR_NUM, comments=comments)

    # Verify MR details were fetched
    details_url = f'{BASE_URL}/merge_requests/{PR_NUM}'
    mock_client_instance.get.assert_called_once_with(
        details_url, headers=gitlab_handler.headers
    )
    mock_get_response.raise_for_status.assert_called_once()

    # Verify discussion was posted
    discussions_url = f'{BASE_URL}/merge_requests/{PR_NUM}/discussions'
    expected_payload = {
        'body': 'GitLab comment',
        'position': {
            'position_type': 'text',
            'base_sha': 'abc',
            'start_sha': 'def',
            'head_sha': 'ghi',
            'new_path': 'src/main.py',
            'new_line': 50,
        },
    }
    mock_client_instance.post.assert_called_once_with(
        discussions_url, headers=gitlab_handler.headers, json=expected_payload
    )
    # Note: We don't check raise_for_status on post because the code only logs non-201


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_gitlab_post_review_no_comments(MockAsyncClient, gitlab_handler):
    """Tests posting a review with no comments to GitLab MR (should not call API)."""
    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value

    await gitlab_handler.post_review(pr_number=PR_NUM, comments=[])

    # Assert that neither get (for details) nor post (for comment) was called
    mock_client_instance.get.assert_not_called()
    mock_client_instance.post.assert_not_called()


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_gitlab_post_review_fetch_details_error(MockAsyncClient, gitlab_handler):
    """Tests posting review when fetching MR details fails."""
    # Mock error response for fetching MR details
    mock_get_response = AsyncMock(spec=httpx.Response)
    mock_get_response.status_code = 404
    mock_get_error = httpx.HTTPStatusError(
        'Not Found', request=MagicMock(), response=mock_get_response
    )
    mock_get_response.raise_for_status = MagicMock(side_effect=mock_get_error)

    # Mock success response for posting the discussion (will still be attempted)
    mock_post_response = AsyncMock(spec=httpx.Response)
    mock_post_response.status_code = 201
    mock_post_response.raise_for_status = MagicMock()

    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
    mock_client_instance.get.return_value = mock_get_response
    mock_client_instance.post.return_value = mock_post_response

    comments = [
        ReviewComment(
            path='src/main.py', line=50, comment='GitLab comment without position'
        )
    ]
    # The function currently logs the error and proceeds without position
    await gitlab_handler.post_review(pr_number=PR_NUM, comments=comments)

    # Verify MR details fetch was attempted
    details_url = f'{BASE_URL}/merge_requests/{PR_NUM}'
    mock_client_instance.get.assert_called_once_with(
        details_url, headers=gitlab_handler.headers
    )
    mock_get_response.raise_for_status.assert_called_once()

    # Verify discussion was posted without position
    discussions_url = f'{BASE_URL}/merge_requests/{PR_NUM}/discussions'
    expected_payload = {
        'body': 'GitLab comment without position',
        # No 'position' key
    }
    mock_client_instance.post.assert_called_once_with(
        discussions_url, headers=gitlab_handler.headers, json=expected_payload
    )


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_gitlab_post_review_post_comment_error(MockAsyncClient, gitlab_handler):
    """Tests posting review when posting the comment itself fails."""
    # Mock success for fetching MR details
    mock_get_response = AsyncMock(spec=httpx.Response)
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = {
        'iid': PR_NUM,
        'target_project_id': 12345,
        'diff_refs': {'base_sha': 'abc', 'start_sha': 'def', 'head_sha': 'ghi'},
    }
    mock_get_response.raise_for_status = MagicMock()

    # Mock error for posting the discussion
    mock_post_response = AsyncMock(spec=httpx.Response)
    mock_post_response.status_code = 400  # Bad Request
    mock_post_response.text = 'Invalid comment format'
    # The code doesn't raise_for_status on post, it just logs non-201
    # So we don't need to mock raise_for_status here for the error case.

    mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
    mock_client_instance.get.return_value = mock_get_response
    mock_client_instance.post.return_value = mock_post_response

    comments = [ReviewComment(path='src/main.py', line=50, comment='Failed comment')]
    # The function currently logs the error and continues
    await gitlab_handler.post_review(pr_number=PR_NUM, comments=comments)

    # Verify MR details fetch
    mock_client_instance.get.assert_called_once()
    mock_get_response.raise_for_status.assert_called_once()

    # Verify discussion post attempt
    mock_client_instance.post.assert_called_once()
    # No raise_for_status check needed here based on current implementation
