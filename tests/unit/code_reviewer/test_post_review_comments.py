import dataclasses
import json
from unittest.mock import AsyncMock, patch

import pytest

from openhands.code_reviewer.post_review_comments import post_comments
from openhands.code_reviewer.reviewer_output import ReviewComment, ReviewerOutput
from openhands.integrations.service_types import ProviderType
from openhands.resolver.interfaces.issue import Issue, IssueHandlerInterface


@pytest.fixture
def sample_review_output():
    return ReviewerOutput(
        pr_info=Issue(
            number=123,
            repo='test/repo',
            owner='test',
            title='Test PR',
            description='A test PR',
            body='Body of test PR',
        ),
        success=True,
        error=None,
        review_level='line',
        review_depth='quick',
        instruction='Review this PR',
        history=[],
        metrics={},
        comments=[
            ReviewComment(path='file1.py', line=10, comment='Comment 1'),
            ReviewComment(path='file2.py', line=20, comment='Comment 2'),
        ],
    )


@patch('openhands.code_reviewer.post_review_comments.get_pr_handler')
def test_post_comments_success(mock_get_handler, tmp_path, sample_review_output):
    """Tests successful posting of comments from a valid JSONL file."""
    mock_handler = AsyncMock(spec=IssueHandlerInterface, post_review=AsyncMock())
    mock_get_handler.return_value = mock_handler

    output_file = tmp_path / 'review_output_123.jsonl'
    # Need to fix the ReviewerOutput structure before dumping
    output_dict = dataclasses.asdict(sample_review_output)
    output_dict['pr_info'] = (
        sample_review_output.pr_info.model_dump()
    )  # Serialize Issue
    output_data = json.dumps(output_dict)

    # Use standard open for writing the test file
    with open(output_file, mode='w') as f:
        f.write(output_data + '\n')  # Write as JSONL

    post_comments(
        str(output_file), token=None, selected_repo='test/repo', pr_number=123
    )

    # Extract owner/repo from selected_repo
    owner, repo_name = 'test', 'repo'
    mock_get_handler.assert_called_once_with(
        owner, repo_name, None, ProviderType.GITHUB, None
    )  # Added base_domain=None
    mock_handler.post_review.assert_called_once_with(
        pr_number=123, comments=sample_review_output.comments
    )


@patch('openhands.code_reviewer.post_review_comments.get_pr_handler')
def test_post_comments_file_not_found(mock_get_handler):
    """Tests behavior when the JSONL file does not exist."""
    mock_handler = AsyncMock(spec=IssueHandlerInterface, post_review=AsyncMock())
    mock_get_handler.return_value = mock_handler

    # post_comments now handles FileNotFoundError internally and logs an error
    post_comments(
        'non_existent_file.jsonl', token=None, selected_repo='test/repo', pr_number=123
    )
    mock_get_handler.assert_not_called()  # Handler shouldn't be created if file not found
    mock_handler.post_review.assert_not_called()


@patch('openhands.code_reviewer.post_review_comments.get_pr_handler')
def test_post_comments_empty_file(mock_get_handler, tmp_path):
    """Tests behavior when the JSONL file is empty."""
    mock_handler = AsyncMock(spec=IssueHandlerInterface, post_review=AsyncMock())
    mock_get_handler.return_value = mock_handler

    output_file = tmp_path / 'empty_review_output.jsonl'
    output_file.touch()  # Create empty file

    # post_comments should handle empty file gracefully (log error)
    post_comments(
        str(output_file), token=None, selected_repo='test/repo', pr_number=123
    )
    mock_get_handler.assert_not_called()  # Handler shouldn't be created if file is empty/invalid
    mock_handler.post_review.assert_not_called()
    # TODO: Add assertion for logging if logging is implemented


@patch('openhands.code_reviewer.post_review_comments.get_pr_handler')
def test_post_comments_invalid_json(mock_get_handler, tmp_path):
    """Tests behavior when the JSONL file contains invalid JSON."""
    mock_handler = AsyncMock(spec=IssueHandlerInterface, post_review=AsyncMock())
    mock_get_handler.return_value = mock_handler

    output_file = tmp_path / 'invalid_json.jsonl'
    with open(output_file, mode='w') as f:
        f.write('this is not valid json\\n')  # Write invalid JSON

    # post_comments should handle JSONDecodeError gracefully (log error)
    post_comments(
        str(output_file), token=None, selected_repo='test/repo', pr_number=123
    )

    mock_get_handler.assert_not_called()  # Handler shouldn't be created if JSON is invalid
    mock_handler.post_review.assert_not_called()
    # TODO: Add assertion for logging if logging is implemented


@patch('openhands.code_reviewer.post_review_comments.get_pr_handler')
def test_post_comments_missing_comments_field(
    mock_get_handler, tmp_path, sample_review_output
):
    """Tests behavior when 'comments' field is null in the JSONL."""
    mock_handler = AsyncMock(spec=IssueHandlerInterface, post_review=AsyncMock())
    mock_get_handler.return_value = mock_handler

    output_file = tmp_path / 'missing_comments.jsonl'
    # Write *something* to the file, the content doesn't matter much as parsing is mocked
    output_data = json.dumps(
        {
            'pr_info': sample_review_output.pr_info.model_dump(),
            'review_level': 'line',
            'review_depth': 'quick',
            'instruction': 'Review this PR',
            'error': None,
            'history': [],
            'metrics': {},
            'success': True,
            'comments': None,  # Explicitly null
        }
    )
    with open(output_file, mode='w') as f:
        f.write(output_data + '\n')

    post_comments(
        str(output_file), token=None, selected_repo='test/repo', pr_number=123
    )

    # Handler should NOT be created if comments are null/missing
    mock_get_handler.assert_not_called()
    mock_handler.post_review.assert_not_called()
    # TODO: Add assertion for logging if logging is implemented


@patch('openhands.code_reviewer.post_review_comments.get_pr_handler')
def test_post_comments_empty_comments_list(
    mock_get_handler, tmp_path, sample_review_output
):
    """Tests behavior when 'comments' list is empty."""
    mock_handler = AsyncMock(spec=IssueHandlerInterface, post_review=AsyncMock())
    mock_get_handler.return_value = mock_handler

    output_file = tmp_path / 'empty_comments_list.jsonl'
    sample_review_output.comments = []  # Set comments to empty list
    # Need to fix the ReviewerOutput structure before dumping
    output_dict = dataclasses.asdict(sample_review_output)
    output_dict['pr_info'] = (
        sample_review_output.pr_info.model_dump()
    )  # Serialize Issue
    # Remove fields not present in the actual JSONL output from review_pr.py

    output_data = json.dumps(output_dict)

    with open(output_file, mode='w') as f:
        f.write(output_data + '\n')

    post_comments(
        str(output_file), token=None, selected_repo='test/repo', pr_number=123
    )

    # Handler should NOT be created if comments are empty
    mock_get_handler.assert_not_called()
    mock_handler.post_review.assert_not_called()
    # TODO: Add assertion for logging if logging is implemented


@patch('openhands.code_reviewer.post_review_comments.get_pr_handler')
def test_post_comments_multiple_lines(mock_get_handler, tmp_path, sample_review_output):
    """Tests posting comments when the JSONL file has multiple lines (should only process first)."""
    mock_handler = AsyncMock(spec=IssueHandlerInterface, post_review=AsyncMock())
    mock_get_handler.return_value = mock_handler

    output_file = tmp_path / 'multiple_lines.jsonl'

    # Prepare first line data
    output_dict1 = dataclasses.asdict(sample_review_output)
    output_dict1['pr_info'] = sample_review_output.pr_info.model_dump()
    output_data1 = json.dumps(output_dict1)

    # Prepare second line data (different comments)
    sample_review_output.comments = [
        ReviewComment(path='file2.py', line=5, comment='Second comment')
    ]
    output_dict2 = dataclasses.asdict(sample_review_output)
    output_dict2['pr_info'] = sample_review_output.pr_info.model_dump()
    output_data2 = json.dumps(output_dict2)

    with open(output_file, mode='w') as f:
        f.write(output_data1 + '\n')
        f.write(output_data2 + '\n')

    post_comments(
        str(output_file), token=None, selected_repo='test/repo', pr_number=123
    )

    # Should only post comments from the first line
    owner, repo_name = 'test', 'repo'
    mock_get_handler.assert_called_once_with(
        owner, repo_name, None, ProviderType.GITHUB, None
    )  # Added base_domain=None
    mock_handler.post_review.assert_called_once_with(
        pr_number=123,
        comments=[
            ReviewComment(path='file1.py', line=10, comment='Comment 1'),
            ReviewComment(path='file2.py', line=20, comment='Comment 2'),
        ],  # Comments from the first line
    )
