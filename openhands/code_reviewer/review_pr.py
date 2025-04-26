import argparse
import asyncio
import dataclasses  # Added for serialization
import json
import os
import pathlib
import shutil
from typing import Any, Dict, List

import aiofiles  # type: ignore[import-untyped]
import httpx
from jinja2 import Template
from pydantic import SecretStr

import openhands

# from openhands.resolver.interfaces.issue_definitions import ServiceContextPR # Removed, not used
from openhands.code_reviewer.reviewer_output import ReviewComment, ReviewerOutput
from openhands.controller.state.state import State  # Added Metrics
from openhands.core.config import AgentConfig, AppConfig, LLMConfig, SandboxConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.core.schema import (
    AgentState,  # Correct import
)
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.event import Event  # Added for history typing
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,  # Added for error checking
    Observation,
)
from openhands.events.stream import EventStreamSubscriber
from openhands.integrations.service_types import ProviderType
from openhands.resolver.interfaces.github import (
    GithubPRHandler,  # Removed GithubIssueHandler
)
from openhands.resolver.interfaces.gitlab import (
    GitlabPRHandler,  # Removed GitlabIssueHandler
)
from openhands.resolver.interfaces.issue import (  # Added IssueHandlerInterface
    Issue,
    IssueHandlerInterface,
)
from openhands.resolver.utils import (
    codeact_user_response,
    get_unique_uid,
    identify_token,
    reset_logger_for_multiprocessing,
)
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import GENERAL_TIMEOUT, call_async_from_sync

# Don't make this confgurable for now, unless we have other competitive agents
AGENT_CLASS = 'CodeActAgent'


def initialize_runtime(
    runtime: Runtime,
    platform: ProviderType,
) -> None:
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    Currently it does nothing.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: Observation

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f'Failed to change directory to /workspace.\n{obs}')

    if platform == ProviderType.GITLAB and os.getenv('GITLAB_CI') == 'true':
        action = CmdRunAction(command='sudo chown -R 1001:0 /workspace/*')
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    action = CmdRunAction(command='git config --global core.pager ""')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f'Failed to set git config.\n{obs}')


async def process_pr_for_review(
    issue: Issue,
    platform: ProviderType,
    # base_commit: str, # Removed, not used here
    max_iterations: int,
    llm_config: LLMConfig,
    output_dir: str,
    base_container_image: str | None,
    runtime_container_image: str | None,
    prompt_template: str,
    issue_handler: IssueHandlerInterface,  # Use interface type hint
    repo_dir: str,
    repo_instruction: str | None = None,
    reset_logger: bool = False,
    review_level: str = 'file',
    review_depth: str = 'quick',
) -> ReviewerOutput:
    # Setup the logger properly, so you can run multi-processing to parallelize processing
    if reset_logger:
        log_dir = os.path.join(output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, str(issue.number), log_dir)
    else:
        logger.info(f'Starting review process for PR {issue.number}.')

    # Define workspace relative to the current directory (GITHUB_WORKSPACE)
    workspace_base = os.path.join(
        '.',  # Current directory
        'workspace',
        f'pr_{issue.number}',
    )
    # Get the absolute path of the workspace base
    workspace_base = os.path.abspath(workspace_base)
    # write the repo to the workspace (assuming repo is already cloned to output_dir/repo)
    if os.path.exists(workspace_base):
        shutil.rmtree(workspace_base)
    # Copy the checked-out repo (from repo_dir) to the workspace
    shutil.copytree(repo_dir, workspace_base)

    sandbox_config = SandboxConfig(
        base_container_image=base_container_image,
        runtime_container_image=runtime_container_image,
        enable_auto_lint=False,
        use_host_network=False,
        timeout=300,
    )

    if os.getenv('GITLAB_CI') == 'true':
        sandbox_config.local_runtime_url = os.getenv(
            'LOCAL_RUNTIME_URL', 'http://localhost'
        )
        user_id = os.getuid() if hasattr(os, 'getuid') else 1000
        if user_id == 0:
            sandbox_config.user_id = get_unique_uid()

    config = AppConfig(
        default_agent='CodeActAgent',
        runtime='docker',
        max_budget_per_task=4,
        max_iterations=max_iterations,
        sandbox=sandbox_config,
        workspace_base=workspace_base,
        workspace_mount_path=workspace_base,
        agents={'CodeActAgent': AgentConfig(disabled_microagents=['github'])},
    )

    config.set_llm_config(llm_config)

    # Prepare the initial prompt/instruction for code review
    template = Template(prompt_template)
    pr_diff = ''
    try:
        # Ensure get_pr_diff exists and call it
        if not hasattr(issue_handler, 'get_pr_diff'):
            raise AttributeError(
                f"{type(issue_handler).__name__} does not have method 'get_pr_diff'"
            )
        pr_diff = await issue_handler.get_pr_diff(issue.number)  # Added await
    except Exception as e:
        logger.error(f'Failed to get PR diff for PR #{issue.number}: {e}')
        return ReviewerOutput(
            pr_info=issue,
            review_level=review_level,
            review_depth=review_depth,
            instruction='',  # No instruction generated
            history=[],
            success=False,
            error=f'Failed to get PR diff: {e}',
        )

    prompt_vars = {
        'issue': issue,
        'repo_instruction': repo_instruction,
        'pr_diff': pr_diff,
        'review_level': review_level,
        'review_depth': review_depth,
    }
    instruction = template.render(prompt_vars)
    logger.info(f'Generated Instruction (first 200 chars): {instruction[:200]}...')

    images_urls: List[str] = []  # Type hint added

    # Initialize variables needed for results
    runtime = None  # Define runtime here to ensure it's available in finally
    event_stream = None
    original_main_subscribers = {}
    state: State | None = None
    comments: List[ReviewComment] = []
    success = False
    error_message: str | None = None
    final_agent_state: AgentState | None = None
    agent_history: List[Event] = []
    agent_metrics: Dict[str, Any] | None = None  # Added from resolve_issue

    # 1. Create and connect runtime
    logger.info('Creating and connecting runtime...')
    runtime = create_runtime(config)
    await runtime.connect()
    logger.info('Runtime connected.')
    event_stream = runtime.event_stream

    # 2. Backup and remove MAIN subscribers (temporary fix for EOFError)
    if event_stream:
        original_main_subscribers = event_stream._subscribers.get(
            EventStreamSubscriber.MAIN, {}
        ).copy()
        if original_main_subscribers:
            logger.info(
                f'Temporarily removing {len(original_main_subscribers)} MAIN subscribers.'
            )
            for callback_id in list(original_main_subscribers.keys()):
                event_stream.unsubscribe(EventStreamSubscriber.MAIN, callback_id)
    else:
        logger.warning('Runtime does not have an event_stream attribute.')

    # 3. Initialize runtime (e.g., git config)
    logger.info('Initializing runtime...')
    initialize_runtime(runtime, platform)
    logger.info('Runtime initialized.')
    # 4. Create initial action and run the agent controller
    action = MessageAction(content=instruction, image_urls=images_urls)
    logger.info(f'Starting agent loop with initial action: {action}')

    try:
        state = await run_controller(
            config=config,
            initial_user_action=action,
            runtime=runtime,
            fake_user_response_fn=codeact_user_response,
        )
        if state is None:
            error_message = 'Agent controller did not return a final state.'
            logger.error(error_message)
            final_agent_state = AgentState.ERROR  # Treat as error
        else:
            final_agent_state = state.agent_state
            agent_history = state.history  # Store history
            agent_metrics = (
                state.metrics.get() if state.metrics else None
            )  # Store metrics
            logger.info(f'Final agent state: {final_agent_state}')

            # Check for errors first
            if final_agent_state == AgentState.ERROR:
                error_message = 'Agent finished in ERROR state.'
                # Try to find a more specific error in history
                if agent_history:
                    for event in reversed(agent_history):
                        if isinstance(event, ErrorObservation):
                            error_message = f'Agent error: {event.content}'
                            break
                logger.error(error_message)
            elif final_agent_state != AgentState.FINISHED:
                error_message = (
                    f'Agent finished in unexpected state: {final_agent_state}'
                )
                logger.warning(
                    error_message
                )  # Log as warning, maybe comments were still generated

            # Attempt to extract comments even if agent didn't finish perfectly
            if agent_history:
                last_event = agent_history[-1]
                if (
                    isinstance(last_event, MessageAction)
                    and last_event.source == 'agent'
                ):
                    try:
                        parsed_comments = json.loads(last_event.content)
                        if isinstance(parsed_comments, list):
                            validated_comments = []
                            for c_dict in parsed_comments:
                                if isinstance(c_dict, dict) and 'comment' in c_dict:
                                    # Validate structure before creating ReviewComment
                                    path = c_dict.get('path')
                                    line = c_dict.get('line')
                                    comment_text = c_dict['comment']
                                    if path is not None and not isinstance(path, str):
                                        logger.warning(
                                            f'Skipping comment with invalid path type: {c_dict}'
                                        )
                                        continue
                                    if line is not None and not isinstance(line, int):
                                        # Try converting to int if it's a string representation
                                        if isinstance(line, str) and line.isdigit():
                                            line = int(line)
                                        else:
                                            logger.warning(
                                                f'Skipping comment with invalid line type: {c_dict}'
                                            )
                                            continue
                                    if not isinstance(comment_text, str):
                                        logger.warning(
                                            f'Skipping comment with invalid comment text type: {c_dict}'
                                        )
                                        continue

                                    validated_comments.append(
                                        ReviewComment(
                                            path=path,
                                            comment=comment_text,
                                            line=line,
                                            # Removed 'level' - not part of ReviewComment
                                        )
                                    )
                                else:
                                    logger.warning(
                                        f'Skipping invalid comment structure: {c_dict}'
                                    )
                            comments = validated_comments
                            logger.info(f'Extracted {len(comments)} review comments.')
                            # If we got comments AND the agent finished, it's a success
                            if final_agent_state == AgentState.FINISHED:
                                success = True
                                error_message = (
                                    None  # Clear any previous warning message
                                )
                        else:
                            parse_error = (
                                "Agent's final message content was not a JSON list."
                            )
                            logger.error(
                                parse_error
                                + f' Content snippet: {last_event.content[:200]}'
                            )
                            if not error_message:
                                error_message = (
                                    parse_error  # Keep original error if agent failed
                                )
                    except json.JSONDecodeError as e:
                        parse_error = (
                            f"Failed to parse agent's final message as JSON: {e}"
                        )
                        logger.error(
                            parse_error
                            + f' Content snippet: {last_event.content[:200]}'
                        )
                        if not error_message:
                            error_message = parse_error
                    except Exception as e:
                        parse_error = f"Error processing agent's final message: {e}"
                        logger.error(
                            parse_error
                            + f' Content snippet: {last_event.content[:200]}'
                        )
                        if not error_message:
                            error_message = parse_error
                elif (
                    not error_message
                ):  # Only set this error if no agent error occurred
                    error_message = f"Agent's final action was not a MessageAction from agent. Last event: {type(last_event).__name__}"
                    logger.error(error_message)
            elif not error_message:  # Only set this error if no agent error occurred
                error_message = 'State history is empty.'
                logger.error(error_message)

            # Final check: if we didn't succeed, ensure there's an error message
            if not success and not error_message:
                error_message = 'Review generation failed for an unknown reason.'
                logger.error(error_message)

    except Exception as e:
        # Catch any other unexpected errors during processing
        logger.exception('An unexpected exception occurred during agent execution:')
        success = False
        comments = []
        error_message = f'Unexpected error during agent execution: {str(e)}'
        final_agent_state = AgentState.ERROR  # Assume error state

    finally:
        # 6. Restore MAIN subscribers
        if event_stream and original_main_subscribers:
            logger.info(f'Restoring {len(original_main_subscribers)} MAIN subscribers.')
            for callback_id, callback_fn in original_main_subscribers.items():
                event_stream.subscribe(
                    EventStreamSubscriber.MAIN, callback_fn, callback_id
                )

        # Ensure runtime is closed if it was created
        if runtime:
            await runtime.close()  # type: ignore[func-returns-value] # runtime.close() returns None

    # Construct the final output
    output = ReviewerOutput(
        pr_info=issue,
        review_level=review_level,
        review_depth=review_depth,
        instruction=instruction,
        history=[
            evt.to_dict() if hasattr(evt, 'to_dict') else dataclasses.asdict(evt)
            for evt in agent_history
        ],  # Serialize history
        comments=comments,
        metrics=agent_metrics,  # Pass metrics
        success=success,
        error=error_message,
    )

    return output


def pr_handler_factory(
    owner: str,
    repo: str,
    token: str,
    # llm_config: LLMConfig, # Removed, not needed here
    platform: ProviderType,
    username: str | None = None,
    base_domain: str | None = None,
) -> IssueHandlerInterface:  # Return interface type
    # Determine default base_domain based on platform
    if base_domain is None:
        base_domain = 'github.com' if platform == ProviderType.GITHUB else 'gitlab.com'

    if platform == ProviderType.GITHUB:
        # Return the handler directly, not wrapped in ServiceContextPR
        return GithubPRHandler(owner, repo, token, username, base_domain)
    elif platform == ProviderType.GITLAB:
        # Return the handler directly, not wrapped in ServiceContextPR
        return GitlabPRHandler(owner, repo, token, username, base_domain)
    else:
        raise ValueError(f'Unsupported platform: {platform}')


async def review_pr_entrypoint(
    owner: str,
    repo: str,
    token: str,
    username: str,
    platform: ProviderType,
    max_iterations: int,
    output_dir: str,
    llm_config: LLMConfig,
    base_container_image: str | None,
    runtime_container_image: str | None,
    prompt_template: str,
    review_level: str,
    review_depth: str,
    repo_instruction: str | None,
    pr_number: int,
    comment_id: int | None,
    reset_logger: bool = False,
    base_domain: str | None = None,
) -> None:
    issue: Issue | None = None

    # Setup output directory and log file early to ensure it exists for error logging
    output_file = os.path.join(output_dir, 'output', f'review_output_{pr_number}.jsonl')
    pathlib.Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    log_dir = os.path.join(output_dir, 'infer_logs')
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f'Using output directory: {output_dir}')
    logger.info(f'Writing output to {output_file}')

    """Review a single pull request.

    Args:
        owner: owner of the repo.
        repo: repository to review PRs in form of `owner/repo`.
        token: token to access the repository.
        username: username to access the repository.
        platform: platform of the repository.
        max_iterations: Maximum number of iterations to run.
        output_dir: Output directory to write the results.
        llm_config: Configuration for the language model.
        base_container_image: Base container image for sandbox.
        runtime_container_image: Runtime container image for sandbox.
        prompt_template: Prompt template to use.
        review_level: Level of review (e.g., 'line', 'file', 'pr').
        review_depth: Depth of review (e.g., 'quick', 'deep').
        repo_instruction: Repository instruction to use.
        pr_number: Pull Request number to review.
        comment_id: Optional ID of a specific comment to focus on.
        reset_logger: Whether to reset the logger for multiprocessing.
        base_domain: The base domain for the git server (defaults to "github.com" for GitHub and "gitlab.com" for GitLab)
    """
    # Determine default base_domain based on platform
    if base_domain is None:
        base_domain = 'github.com' if platform == ProviderType.GITHUB else 'gitlab.com'

    try:
        pr_handler = pr_handler_factory(
            owner, repo, token, platform, username, base_domain
        )

        # Load PR data
        prs: list[Issue] = pr_handler.get_converted_issues(
            issue_numbers=[pr_number], comment_id=comment_id
        )

        if not prs:
            raise ValueError(
                f'No PR found for PR number {pr_number}. Please verify that:\n'
                f'1. The PR #{pr_number} exists in the repository {owner}/{repo}\n'
                f'2. You have the correct permissions to access it\n'
                f'3. The repository name is spelled correctly'
            )

        pr_info = prs[0]

        if comment_id is not None:
            # Check if the provided comment_id actually exists in the fetched PR data
            all_comments = (
                (pr_info.review_comments or [])
                + (pr_info.issue_comments or [])
                + (
                    pr_info.review_threads or []
                )  # Assuming review_threads contain comments
            )
            # Attempt to find the comment ID, converting to string for comparison
            found_comment = False
            for comment in all_comments:
                if comment and str(comment.get('id', '')) == str(comment_id):
                    found_comment = True
                    break
            if not found_comment:
                logger.warning(
                    f'Comment ID {comment_id} provided, but no matching comment found for PR #{pr_number}. Proceeding with full PR review.'
                )
                # Reset comment_id so the agent doesn't focus on a non-existent comment
                comment_id = None

        # Assume repository is already cloned and checked out to the correct state
        # by the CI/CD workflow in the `output_dir/repo` directory.
        repo_dir = os.environ.get('GITHUB_WORKSPACE')
        if not repo_dir or not os.path.exists(os.path.join(repo_dir, '.git')):
            raise FileNotFoundError(
                f'Repository not found or not a git repository in GITHUB_WORKSPACE ({repo_dir}). Please ensure the workflow checks out the repo.'
            )

        # Load repo-specific instructions if not provided via args
        if repo_instruction is None:
            guideline_path_md = os.path.join(
                repo_dir, '.github', 'CODE_REVIEW_GUIDELINES.md'
            )
            guideline_path_txt = os.path.join(
                repo_dir, '.github', 'CODE_REVIEW_GUIDELINES.txt'
            )
            openhands_instructions_path = os.path.join(
                repo_dir, '.openhands_instructions'
            )
            instruction_path_to_use = None
            if os.path.exists(guideline_path_md):
                instruction_path_to_use = guideline_path_md
            elif os.path.exists(guideline_path_txt):
                instruction_path_to_use = guideline_path_txt
            elif os.path.exists(openhands_instructions_path):
                instruction_path_to_use = openhands_instructions_path

            if instruction_path_to_use:
                logger.info(
                    f'Using repository instruction file: {instruction_path_to_use}'
                )
                try:
                    async with aiofiles.open(instruction_path_to_use, mode='r') as f:
                        repo_instruction = await f.read()
                except Exception as e:
                    logger.error(f'Error reading repository instruction file: {e}')
                    # Continue without repo instructions if file reading fails

        # Process the PR
        output = await process_pr_for_review(
            issue=pr_info,
            platform=platform,
            # base_commit=base_commit, # Removed
            max_iterations=max_iterations,
            llm_config=llm_config,
            output_dir=output_dir,
            repo_dir=repo_dir,
            base_container_image=base_container_image,
            runtime_container_image=runtime_container_image,
            prompt_template=prompt_template,
            issue_handler=pr_handler,  # Pass the handler instance
            repo_instruction=repo_instruction,
            reset_logger=reset_logger,
            review_level=review_level,
            review_depth=review_depth,
        )

    except (ValueError, AttributeError, FileNotFoundError) as e:
        logger.error(f'Error during setup or PR processing: {e}')
        # Create a basic error output if we failed before processing
        issue_to_log = issue  # Use the 'issue' variable from the outer scope
        if issue_to_log is None:
            try:
                # Try to create a basic Issue object if owner/repo/pr_number are defined
                issue_to_log = Issue(
                    owner=owner,
                    repo=repo,
                    number=pr_number,
                    title=f'PR #{pr_number}',
                    body='',
                )
            except NameError:
                # If owner/repo/pr_number are not defined (error happened very early), create a dummy issue
                issue_to_log = Issue(
                    owner='unknown',
                    repo='unknown',
                    number=pr_number if 'pr_number' in locals() else -1,
                    title=f"PR #{pr_number if 'pr_number' in locals() else 'unknown'}",
                    body='',
                )
        output = ReviewerOutput(
            pr_info=issue_to_log,
            review_level=review_level,
            review_depth=review_depth,
            instruction='',
            history=[],
            success=False,
            error=str(e),
            metrics=None,
            comments=[],
        )
    except httpx.HTTPStatusError as e:
        logger.error(f'HTTP Status Error: {e}')
        logger.error(f'Response body: {e.response.text}')
        # Re-raise the exception after logging
        raise
    except Exception as e:
        logger.exception(
            f'Unexpected error during review_pr_entrypoint for PR {pr_number}:'
        )
        issue_to_log = issue  # Use the 'issue' variable from the outer scope
        if issue_to_log is None:
            try:
                # Try to create a basic Issue object if owner/repo/pr_number are defined
                issue_to_log = Issue(
                    owner=owner,
                    repo=repo,
                    number=pr_number,
                    title=f'PR #{pr_number}',
                    body='',
                )
            except NameError:
                # If owner/repo/pr_number are not defined (error happened very early), create a dummy issue
                issue_to_log = Issue(
                    owner='unknown',
                    repo='unknown',
                    number=pr_number if 'pr_number' in locals() else -1,
                    title=f"PR #{pr_number if 'pr_number' in locals() else 'unknown'}",
                    body='',
                )
        output = ReviewerOutput(
            pr_info=issue_to_log,
            review_level=review_level,
            review_depth=review_depth,
            instruction='',
            history=[],
            success=False,
            error=f'Unexpected error: {str(e)}',
            metrics=None,
            comments=[],
        )

    # Write the output to a JSONL file (ensure output is not None)
    if output is not None:
        output_file = os.path.join(output_dir, f'review_output_{pr_number}.jsonl')
        try:
            async with aiofiles.open(output_file, mode='w') as f:
                # Convert ReviewerOutput to dict, handling nested dataclasses and complex types
                def default_serializer(obj):
                    if hasattr(obj, 'to_dict'):
                        # Use to_dict if available (like for Event subclasses)
                        return obj.to_dict()
                    if dataclasses.is_dataclass(obj):
                        # Use asdict for other dataclasses
                        return dataclasses.asdict(obj)
                    # Add handling for other non-serializable types if necessary
                    try:
                        # Attempt default serialization first (might work for simple types)
                        # Check if it's basic type before encoding
                        if isinstance(
                            obj, (str, int, float, bool, list, dict, type(None))
                        ):
                            return obj
                        return str(obj)  # Fallback to string representation
                    except TypeError:
                        return str(obj)  # Final fallback

                # Use dataclasses.asdict for the main object, then serialize with custom handler
                output_dict = dataclasses.asdict(output)
                await f.write(
                    json.dumps(output_dict, default=default_serializer) + '\n'
                )
            logger.info(f'Review output written to {output_file}')
        except Exception as e:
            logger.error(f'Failed to write output file {output_file}: {e}')


def main() -> None:
    def int_or_none(value: str) -> int | None:
        if value.lower() == 'none':
            return None
        else:
            return int(value)

    parser = argparse.ArgumentParser(description='Review a single pull request.')
    parser.add_argument(
        '--selected-repo',
        type=str,
        required=True,
        help='repository to review PRs in form of `owner/repo`.',
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='token to access the repository.',
    )
    parser.add_argument(
        '--username',
        type=str,
        default=None,
        help='username to access the repository.',
    )
    parser.add_argument(
        '--base-container-image',
        type=str,
        default=None,
        help='base container image to use.',
    )
    parser.add_argument(
        '--runtime-container-image',
        type=str,
        default=None,
        help='Container image to use.',
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,  # Reduced default iterations for review?
        help='Maximum number of iterations to run.',
    )
    parser.add_argument(
        '--pr-number',  # Renamed from --issue-number
        type=int,
        required=True,
        help='Pull Request number to review.',
    )
    parser.add_argument(
        '--comment-id',
        type=int_or_none,
        required=False,
        default=None,
        help='Review a specific comment thread within the PR',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory to write the results.',
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default=None,
        help='LLM model to use.',
    )
    parser.add_argument(
        '--llm-api-key',
        type=str,
        default=None,
        help='LLM API key to use.',
    )
    parser.add_argument(
        '--llm-base-url',
        type=str,
        default=None,
        help='LLM base URL to use.',
    )
    parser.add_argument(
        '--prompt-file',
        type=str,
        default=None,
        help='Path to the prompt template file in Jinja format.',
    )
    parser.add_argument(
        '--repo-instruction-file',
        type=str,
        default=None,
        help='Path to the repository instruction/guideline file in text format.',
    )
    parser.add_argument(
        '--review-level',  # Added
        type=str,
        default='file',
        choices=['line', 'file', 'pr'],
        help='Level of detail for the review (line, file, or overall PR).',
    )
    parser.add_argument(
        '--review-depth',  # Added
        type=str,
        default='quick',
        choices=['quick', 'medium', 'deep'],
        help='Depth/thoroughness of the review (quick, medium, or deep).',
    )
    parser.add_argument(
        '--is-experimental',
        type=lambda x: x.lower() == 'true',
        default=False,
        help='Whether to run in experimental mode.',
    )
    parser.add_argument(
        '--base-domain',
        type=str,
        default=None,
        help='Base domain for the git server (defaults to "github.com" for GitHub and "gitlab.com" for GitLab)',
    )

    my_args = parser.parse_args()

    # Initialize container image variables
    base_container_image: str | None = None
    runtime_container_image: str | None = None
    # Get container image from environment variable first
    env_base_image_as_runtime = os.getenv(
        'SANDBOX_BASE_CONTAINER_IMAGE'
    )  # Check for base image env var to use as runtime

    if env_base_image_as_runtime:
        logger.info(
            f'Using SANDBOX_BASE_CONTAINER_IMAGE as runtime image: {env_base_image_as_runtime}'
        )
        runtime_container_image = env_base_image_as_runtime
        base_container_image = (
            None  # Ensure base_container_image is None if env var is used
        )
    else:
        # Fallback to command-line arguments if env var is not set
        logger.info(
            'SANDBOX_BASE_CONTAINER_IMAGE not set, checking command-line arguments for runtime/base images.'
        )
        arg_base_image = my_args.base_container_image
        arg_runtime_image = my_args.runtime_container_image

        if arg_runtime_image is not None and arg_base_image is not None:
            raise ValueError(
                'Cannot provide both --runtime-container-image and --base-container-image via arguments when SANDBOX_BASE_CONTAINER_IMAGE is not set.'
            )

        # Determine the final image configuration based on args
        if arg_base_image is not None:
            logger.info(f'Using base container image from args: {arg_base_image}')
            base_container_image = arg_base_image
            # runtime_container_image remains None
        elif arg_runtime_image is not None:
            logger.info(f'Using runtime container image from args: {arg_runtime_image}')
            runtime_container_image = arg_runtime_image
            # base_container_image remains None
        elif not my_args.is_experimental:
            # Neither arg provided, not experimental: use default runtime image
            runtime_container_image = (
                f'ghcr.io/all-hands-ai/runtime:{openhands.__version__}-nikolaik'
            )
            logger.info(
                f'Defaulting runtime container image to: {runtime_container_image}'
            )
            # base_container_image remains None
        else:
            # Neither arg provided, IS experimental: leave both as None
            logger.info(
                'No container image specified via args or env, and is_experimental=True. Both images remain None.'
            )
            # Both base_container_image and runtime_container_image remain None

    parts = my_args.selected_repo.rsplit('/', 1)
    if len(parts) < 2:
        raise ValueError('Invalid repository format. Expected owner/repo')
    owner, repo = parts

    token_str = my_args.token or os.getenv('GITHUB_TOKEN') or os.getenv('GITLAB_TOKEN')
    username = my_args.username if my_args.username else os.getenv('GIT_USERNAME')
    if not username:
        raise ValueError('Username is required.')

    if not token_str:
        raise ValueError('Token is required.')

    token = token_str

    platform = call_async_from_sync(
        identify_token,
        GENERAL_TIMEOUT,
        token,
        my_args.base_domain,
    )

    api_key = my_args.llm_api_key or os.environ['LLM_API_KEY']
    model = my_args.llm_model or os.environ['LLM_MODEL']
    base_url = my_args.llm_base_url or os.environ.get('LLM_BASE_URL', None)
    api_version = os.environ.get('LLM_API_VERSION', None)

    # Create LLMConfig instance
    llm_config = LLMConfig(
        model=model,
        api_key=SecretStr(api_key) if api_key else None,
        base_url=base_url,
    )

    # Only set api_version if it was explicitly provided, otherwise let LLMConfig handle it
    if api_version is not None:
        llm_config.api_version = api_version

    repo_instruction = None
    if my_args.repo_instruction_file:
        with open(my_args.repo_instruction_file, 'r') as f:
            repo_instruction = f.read()

    # Set default prompt file if not provided
    prompt_file = my_args.prompt_file
    if prompt_file is None:
        # Use a default review prompt (adjust path as needed)
        prompt_file = os.path.join(
            os.path.dirname(__file__), 'prompts/review/basic-review.jinja'
        )
        logger.info(f'Prompt file not specified, using default: {prompt_file}')

    # Read the prompt template
    try:
        with open(prompt_file, 'r') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        logger.error(f'Prompt template file not found: {prompt_file}')
        raise
    except Exception as e:
        logger.error(f'Error reading prompt template file {prompt_file}: {e}')
        raise

    asyncio.run(
        review_pr_entrypoint(  # Changed from resolve_issue
            owner=owner,
            repo=repo,
            token=token,
            username=username,
            platform=platform,
            base_container_image=base_container_image,
            runtime_container_image=runtime_container_image,
            max_iterations=my_args.max_iterations,
            output_dir=my_args.output_dir,
            llm_config=llm_config,
            prompt_template=prompt_template,
            review_level=my_args.review_level,  # Added
            review_depth=my_args.review_depth,  # Added
            repo_instruction=repo_instruction,
            pr_number=my_args.pr_number,  # Changed from issue_number
            comment_id=my_args.comment_id,
            base_domain=my_args.base_domain,
        )
    )


if __name__ == '__main__':
    main()
