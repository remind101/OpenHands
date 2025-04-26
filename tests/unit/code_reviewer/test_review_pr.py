import dataclasses  # Added import
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.code_reviewer.review_pr import process_pr_for_review
from openhands.code_reviewer.reviewer_output import ReviewComment
from openhands.controller.state.state import State
from openhands.core.config import LLMConfig
from openhands.core.schema import AgentState
from openhands.events.action import MessageAction
from openhands.integrations.service_types import ProviderType
from openhands.resolver.interfaces.issue import Issue, IssueHandlerInterface

# Sample data
SAMPLE_ISSUE = Issue(
    number=101,
    repo='test/repo',
    owner='test',
    title='Test PR for Review',
    body='PR body content.',
)
SAMPLE_LLM_CONFIG = LLMConfig(model='gpt-4o')
SAMPLE_PROMPT_TEMPLATE = 'Review this diff:\n{{ pr_diff }}'
SAMPLE_DIFF = 'diff --git a/file.py b/file.py\nindex 123..456 100644\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old line\n+new line\n'


@pytest.fixture
def mock_issue_handler():
    handler = AsyncMock(spec=IssueHandlerInterface)
    handler.get_pr_diff = AsyncMock(return_value=SAMPLE_DIFF)
    # Add other methods if needed by the function under test
    return handler


@pytest.fixture
def setup_workspace(tmp_path):
    output_dir = tmp_path / 'review_output'
    output_dir.mkdir()
    repo_dir = output_dir / 'repo'
    repo_dir.mkdir()
    # Create a dummy file in the repo to be copied
    (repo_dir / 'dummy_file.txt').touch()
    return str(output_dir)


# Mock runtime and controller
@patch('openhands.code_reviewer.review_pr.create_runtime')
@patch('openhands.code_reviewer.review_pr.run_controller')
@pytest.mark.asyncio
async def test_process_pr_success(
    mock_run_controller, mock_create_runtime, mock_issue_handler, setup_workspace
):
    """Tests the success case where the agent finishes and returns valid comments."""
    output_dir = setup_workspace

    # Mock Runtime
    mock_runtime_instance = AsyncMock()
    mock_runtime_instance.run_action = MagicMock()  # Initialize run_action mock
    mock_runtime_instance.close = AsyncMock()
    mock_create_runtime.return_value = mock_runtime_instance

    # Mock Controller State (Success)
    final_state = State()
    final_state.agent_state = AgentState.FINISHED
    final_comments = [
        ReviewComment(path='file.py', line=1, comment='Looks good!'),
        ReviewComment(path='other.py', comment='General comment'),  # No line
    ]
    final_message_content = json.dumps([dataclasses.asdict(c) for c in final_comments])
    final_state.history.append(MessageAction(content=final_message_content))
    final_state.history[-1]._source = 'agent'  # Set internal _source attribute
    mock_run_controller.return_value = final_state

    # Call the function
    result = await process_pr_for_review(
        issue=SAMPLE_ISSUE,
        platform=ProviderType.GITHUB,
        max_iterations=5,
        llm_config=SAMPLE_LLM_CONFIG,
        output_dir=output_dir,
        base_container_image=None,
        runtime_container_image=None,
        prompt_template=SAMPLE_PROMPT_TEMPLATE,
        issue_handler=mock_issue_handler,
        review_level='line',
        review_depth='full',
    )

    # Assertions
    assert result.success is True
    assert result.error is None
    assert result.pr_info == SAMPLE_ISSUE
    assert result.review_level == 'line'
    assert result.review_depth == 'full'
    assert len(result.comments) == 2
    assert result.comments[0].path == 'file.py'
    assert result.comments[0].line == 1
    assert result.comments[0].comment == 'Looks good!'
    assert result.comments[1].path == 'other.py'
    assert result.comments[1].line is None
    assert result.comments[1].comment == 'General comment'
    assert result.history is not None  # Check history exists
    assert result.metrics is not None  # Check metrics exist

    # Verify mocks
    mock_issue_handler.get_pr_diff.assert_awaited_once_with(SAMPLE_ISSUE.number)
    mock_create_runtime.assert_called_once()
    mock_run_controller.assert_awaited_once()
    mock_runtime_instance.close.assert_awaited_once()

    # Check workspace was created
    workspace_path = os.path.join(output_dir, 'workspace', f'pr_{SAMPLE_ISSUE.number}')
    assert os.path.exists(workspace_path)
    assert os.path.exists(os.path.join(workspace_path, 'dummy_file.txt'))


# TODO: Add more test cases: success no comments, agent error, json error, diff error etc.


# Mock runtime and controller
@patch('openhands.code_reviewer.review_pr.create_runtime')
@patch('openhands.code_reviewer.review_pr.run_controller')
@pytest.mark.asyncio
async def test_process_pr_success_no_comments(
    mock_run_controller, mock_create_runtime, mock_issue_handler, setup_workspace
):
    """Tests the success case where the agent finishes but returns no valid comments."""
    output_dir = setup_workspace

    # Mock Runtime
    mock_runtime_instance = AsyncMock()
    mock_runtime_instance.run_action = MagicMock()
    mock_runtime_instance.close = AsyncMock()
    mock_create_runtime.return_value = mock_runtime_instance

    # Mock Controller State (Success, but no valid comments in last message)
    final_state = State()
    final_state.agent_state = AgentState.FINISHED
    # Case 1: Last message is not from agent
    # final_state.history.append(MessageAction(content='User message', source='user'))
    # Case 2: Last message is from agent, but empty content
    # final_state.history.append(MessageAction(content='', source='agent'))
    # Case 3: Last message is from agent, but not valid JSON
    # final_state.history.append(MessageAction(content='Not JSON', source='agent'))
    # Case 4: Last message is from agent, valid JSON, but empty list
    final_state.history.append(MessageAction(content='[]'))
    final_state.history[-1]._source = 'agent'  # Set internal _source attribute
    # Case 5: Last message is from agent, valid JSON, but wrong structure
    # final_state.history.append(MessageAction(content='{"comment": "hello"}', source='agent'))

    mock_run_controller.return_value = final_state

    # Call the function
    result = await process_pr_for_review(
        issue=SAMPLE_ISSUE,
        platform=ProviderType.GITHUB,
        max_iterations=5,
        llm_config=SAMPLE_LLM_CONFIG,
        output_dir=output_dir,
        base_container_image=None,
        runtime_container_image=None,
        prompt_template=SAMPLE_PROMPT_TEMPLATE,
        issue_handler=mock_issue_handler,
        review_level='line',
        review_depth='full',
    )

    # Assertions
    assert result.success is True  # Still successful as agent finished
    assert result.error is None
    assert result.pr_info == SAMPLE_ISSUE
    assert len(result.comments) == 0  # No comments extracted
    assert result.history is not None
    assert result.metrics is not None

    # Verify mocks
    mock_issue_handler.get_pr_diff.assert_awaited_once_with(SAMPLE_ISSUE.number)
    mock_create_runtime.assert_called_once()
    mock_run_controller.assert_awaited_once()
    mock_runtime_instance.close.assert_awaited_once()


# Mock runtime and controller
@patch('openhands.code_reviewer.review_pr.create_runtime')
@patch('openhands.code_reviewer.review_pr.run_controller')
@pytest.mark.asyncio
async def test_process_pr_agent_error(
    mock_run_controller, mock_create_runtime, mock_issue_handler, setup_workspace
):
    """Tests the case where the agent finishes in an error state."""
    output_dir = setup_workspace

    # Mock Runtime
    mock_runtime_instance = AsyncMock()
    mock_runtime_instance.run_action = MagicMock()
    mock_runtime_instance.close = AsyncMock()
    mock_create_runtime.return_value = mock_runtime_instance

    # Mock Controller State (Agent Failed)
    final_state = State()
    final_state.agent_state = AgentState.ERROR
    final_state.history.append(MessageAction(content='Error occurred'))
    final_state.history[-1]._source = 'agent'
    mock_run_controller.return_value = final_state

    # Call the function
    result = await process_pr_for_review(
        issue=SAMPLE_ISSUE,
        platform=ProviderType.GITHUB,
        max_iterations=5,
        llm_config=SAMPLE_LLM_CONFIG,
        output_dir=output_dir,
        base_container_image=None,
        runtime_container_image=None,
        prompt_template=SAMPLE_PROMPT_TEMPLATE,
        issue_handler=mock_issue_handler,
        review_level='line',
        review_depth='full',
    )

    # Assertions
    assert result.success is False
    assert result.error is not None
    assert 'Agent finished in ERROR state.' in result.error
    assert result.pr_info == SAMPLE_ISSUE
    assert len(result.comments) == 0
    assert result.history is not None
    assert result.metrics is not None

    # Verify mocks
    mock_issue_handler.get_pr_diff.assert_awaited_once_with(SAMPLE_ISSUE.number)
    mock_create_runtime.assert_called_once()
    mock_run_controller.assert_awaited_once()
    mock_runtime_instance.close.assert_awaited_once()


# Mock runtime and controller
@patch('openhands.code_reviewer.review_pr.create_runtime')
@patch('openhands.code_reviewer.review_pr.run_controller')
@pytest.mark.asyncio
async def test_process_pr_json_error(
    mock_run_controller, mock_create_runtime, mock_issue_handler, setup_workspace
):
    """Tests the case where the agent finishes but the last message is not valid JSON."""
    output_dir = setup_workspace

    # Mock Runtime
    mock_runtime_instance = AsyncMock()
    mock_runtime_instance.run_action = MagicMock()
    mock_runtime_instance.close = AsyncMock()
    mock_create_runtime.return_value = mock_runtime_instance

    # Mock Controller State (Success, but invalid JSON)
    final_state = State()
    final_state.agent_state = AgentState.FINISHED
    final_state.history.append(MessageAction(content='This is not JSON'))
    final_state.history[-1]._source = 'agent'
    mock_run_controller.return_value = final_state

    # Call the function
    result = await process_pr_for_review(
        issue=SAMPLE_ISSUE,
        platform=ProviderType.GITHUB,
        max_iterations=5,
        llm_config=SAMPLE_LLM_CONFIG,
        output_dir=output_dir,
        base_container_image=None,
        runtime_container_image=None,
        prompt_template=SAMPLE_PROMPT_TEMPLATE,
        issue_handler=mock_issue_handler,
        review_level='line',
        review_depth='full',
    )

    # Assertions
    assert result.success is False  # Agent finished, but comment parsing failed
    assert result.error is not None  # Error should indicate JSON parsing failure
    assert "Failed to parse agent's final message as JSON" in result.error
    assert result.pr_info == SAMPLE_ISSUE
    assert len(result.comments) == 0  # No comments extracted due to JSON error
    assert result.history is not None
    assert result.metrics is not None

    # Verify mocks
    mock_issue_handler.get_pr_diff.assert_awaited_once_with(SAMPLE_ISSUE.number)
    mock_create_runtime.assert_called_once()
    mock_run_controller.assert_awaited_once()
    mock_runtime_instance.close.assert_awaited_once()


# Mock runtime and controller
@patch('openhands.code_reviewer.review_pr.create_runtime')
@patch('openhands.code_reviewer.review_pr.run_controller')
@pytest.mark.asyncio
async def test_process_pr_diff_error(
    mock_run_controller, mock_create_runtime, mock_issue_handler, setup_workspace
):
    """Tests the case where fetching the PR diff fails."""
    output_dir = setup_workspace

    # Mock Issue Handler (Error)
    mock_issue_handler.get_pr_diff.side_effect = Exception('Failed to fetch diff')

    # Mock Runtime (Should not be created)
    mock_runtime_instance = AsyncMock()
    mock_create_runtime.return_value = mock_runtime_instance

    # Mock Controller (Should not be run)
    mock_run_controller.return_value = State()  # Dummy state

    # Call the function
    result = await process_pr_for_review(
        issue=SAMPLE_ISSUE,
        platform=ProviderType.GITHUB,
        max_iterations=5,
        llm_config=SAMPLE_LLM_CONFIG,
        output_dir=output_dir,
        base_container_image=None,
        runtime_container_image=None,
        prompt_template=SAMPLE_PROMPT_TEMPLATE,
        issue_handler=mock_issue_handler,
        review_level='line',
        review_depth='full',
    )

    # Assertions
    assert result.success is False
    assert result.error is not None
    assert 'Failed to fetch diff' in result.error
    assert result.pr_info == SAMPLE_ISSUE
    assert len(result.comments) == 0
    assert result.history == []  # History is initialized but empty
    assert result.metrics is None  # Metrics are part of state

    # Verify mocks
    mock_issue_handler.get_pr_diff.assert_awaited_once_with(SAMPLE_ISSUE.number)
    mock_create_runtime.assert_called_once()  # Runtime is created before diff is fetched
    mock_run_controller.assert_not_awaited()
    mock_runtime_instance.close.assert_awaited_once()  # Runtime should be closed in finally block


# Mock runtime and controller
@patch('openhands.code_reviewer.review_pr.create_runtime')
@patch('openhands.code_reviewer.review_pr.run_controller')
@pytest.mark.asyncio
async def test_process_pr_runtime_error(
    mock_run_controller, mock_create_runtime, mock_issue_handler, setup_workspace
):
    """Tests the case where creating the runtime fails."""
    output_dir = setup_workspace

    # Mock Issue Handler (Success)
    mock_issue_handler.get_pr_diff.return_value = SAMPLE_DIFF

    # Mock Runtime Creation (Error)
    mock_create_runtime.side_effect = Exception('Runtime creation failed')

    # Mock Controller (Should not be run)
    mock_run_controller.return_value = State()  # Dummy state

    # Call the function
    result = await process_pr_for_review(
        issue=SAMPLE_ISSUE,
        platform=ProviderType.GITHUB,
        max_iterations=5,
        llm_config=SAMPLE_LLM_CONFIG,
        output_dir=output_dir,
        base_container_image=None,
        runtime_container_image=None,
        prompt_template=SAMPLE_PROMPT_TEMPLATE,
        issue_handler=mock_issue_handler,
        review_level='line',
        review_depth='full',
    )

    # Assertions
    assert result.success is False
    assert result.error is not None
    assert 'Runtime creation failed' in result.error
    assert result.pr_info == SAMPLE_ISSUE
    assert len(result.comments) == 0
    assert result.history == []
    assert result.metrics is None

    # Verify mocks
    mock_issue_handler.get_pr_diff.assert_not_awaited()  # Should not be called if runtime fails
    mock_create_runtime.assert_called_once()
    mock_run_controller.assert_not_awaited()


# Mock runtime and controller
@patch('openhands.code_reviewer.review_pr.create_runtime')
@patch('openhands.code_reviewer.review_pr.run_controller')
@pytest.mark.asyncio
async def test_process_pr_controller_error(
    mock_run_controller, mock_create_runtime, mock_issue_handler, setup_workspace
):
    """Tests the case where the controller fails during execution."""
    output_dir = setup_workspace

    # Mock Issue Handler (Success)
    mock_issue_handler.get_pr_diff.return_value = SAMPLE_DIFF

    # Mock Runtime (Success)
    mock_runtime_instance = AsyncMock()
    mock_runtime_instance.run_action = MagicMock()
    mock_runtime_instance.close = AsyncMock()
    mock_create_runtime.return_value = mock_runtime_instance

    # Mock Controller (Error)
    mock_run_controller.side_effect = Exception('Controller failed')

    # Call the function
    result = await process_pr_for_review(
        issue=SAMPLE_ISSUE,
        platform=ProviderType.GITHUB,
        max_iterations=5,
        llm_config=SAMPLE_LLM_CONFIG,
        output_dir=output_dir,
        base_container_image=None,
        runtime_container_image=None,
        prompt_template=SAMPLE_PROMPT_TEMPLATE,
        issue_handler=mock_issue_handler,
        review_level='line',
        review_depth='full',
    )

    # Assertions
    assert result.success is False
    assert result.error is not None
    assert 'Controller failed' in result.error
    assert result.pr_info == SAMPLE_ISSUE
    assert len(result.comments) == 0
    assert result.history == []  # Controller failed before returning state
    assert result.metrics is None

    # Verify mocks
    mock_issue_handler.get_pr_diff.assert_awaited_once_with(SAMPLE_ISSUE.number)
    mock_create_runtime.assert_called_once()
    mock_run_controller.assert_awaited_once()
    mock_runtime_instance.close.assert_awaited_once()  # Should still be closed in finally block
