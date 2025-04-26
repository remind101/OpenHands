import argparse
import asyncio
import dataclasses  # Added for serialization
import json
import os
import shutil
from typing import Any, Callable, Dict, List, cast  # Add Callable, cast
from uuid import uuid4

import aiofiles  # type: ignore[import-untyped]
from jinja2 import Template
from pydantic import BaseModel, SecretStr

import openhands

# from openhands.resolver.interfaces.issue_definitions import ServiceContextPR # Removed, not used
from openhands.code_reviewer.reviewer_output import ReviewComment, ReviewerOutput
from openhands.controller.state.state import State  # Added Metrics
from openhands.core.config import AgentConfig, AppConfig, LLMConfig, SandboxConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import FakeUserResponseFunc, create_runtime, run_controller
from openhands.core.schema.agent import (
    AgentState,  # Correct import
)
from openhands.events.action import (
    Action,  # Import Action
    AgentFinishAction,
    CmdRunAction,
    MessageAction,
)
from openhands.events.event import (
    Event,  # Added for history typing
)
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
from openhands.resolver.interfaces.issue import (
    Issue,
    IssueHandlerInterface,
)
from openhands.resolver.utils import (
    get_unique_uid,
    identify_token,
    reset_logger_for_multiprocessing,
)
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import GENERAL_TIMEOUT, call_async_from_sync


def handle_awaiting_input(
    current_state: State,  # Change AgentState to State
    encapsulate_solution: bool = False,  # Add optional args
    try_parse: Callable[[Action | None], str] | None = None,  # Add optional args
) -> str:  # Change return type to str
    """Handles the AWAITING_USER_INPUT state by returning a message to finish."""
    logger.info('Agent entered AWAITING_USER_INPUT state. Returning FINISH message.')
    # We instruct the agent to finish, as it should not be waiting for input.
    return 'You should not be waiting for input. Please finalize your review and call the `finish` tool with the JSON list of comments as the `message` argument, as per the instructions.'


# Helper for JSON serialization
def default_serializer(obj):
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    try:
        if isinstance(obj, (str, int, float, bool, list, dict, type(None))):
            return obj
        return str(obj)
    except TypeError:
        return str(obj)


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


async def process_review(
    pr_data: dict[str, Any],  # Changed from issue: Issue
    platform: ProviderType,
    # base_commit: str, # Removed, not used here
    max_iterations: int,
    llm_config: LLMConfig,
    output_dir: str,
    base_container_image: str | None,
    runtime_container_image: str | None,
    prompt_template: str,
    repo_dir: str,
    repo_instruction: str | None = None,
    reset_logger: bool = False,
    review_level: str = 'file',  # Default reverted to file
    review_depth: str = 'quick',  # Default reverted to quick
) -> ReviewerOutput:
    # Setup the logger properly, so you can run multi-processing to parallelize processing
    if reset_logger:
        log_dir = os.path.join(output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, str(pr_data['number']), log_dir)
    else:
        logger.info(f"Starting review process for PR {pr_data['number']}.")

    # Define workspace relative to the current directory (GITHUB_WORKSPACE)
    workspace_base = os.path.join(
        '.',  # Current directory
        'workspace',
        f"pr_{pr_data['number']}",
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
    prompt_vars = {
        'pr_data': pr_data,  # Pass the dictionary
        'repo_instruction': repo_instruction,
        'review_level': review_level,
        'review_depth': review_depth,
    }
    instruction = template.render(prompt_vars)
    logger.info(f'Generated Instruction (first 200 chars): {instruction[:200]}...')

    images_urls: List[str] = []  # Type hint added

    # Initialize variables needed for results
    runtime = None  # Define runtime here to ensure it's available in finally
    event_stream = None
    state: State | None = None
    comments: List[ReviewComment] = []
    success = False
    error_message: str | None = None
    final_agent_state: AgentState | None = None
    agent_history: List[Event] = []
    agent_metrics: Dict[str, Any] | None = None  # Added from resolve_issue

    def on_event(evt: Event) -> None:
        if isinstance(evt, CmdOutputObservation):
            # Log command output observations with truncated content
            MAX_LEN = 200
            content_preview = evt.content[:MAX_LEN]
            if len(evt.content) > MAX_LEN:
                content_preview += '... [truncated]'
            logger.info(
                f"CmdOutputObservation(command={evt.command}, exit_code={evt.exit_code}, content='{content_preview}')"
            )
        else:
            # Log other events normally (might still truncate based on default logger settings)
            logger.info(evt)

    # 1. Create and connect runtime
    logger.info('Creating and connecting runtime...')
    runtime = create_runtime(config)
    await runtime.connect()
    logger.info('Runtime connected.')
    event_stream = runtime.event_stream
    if event_stream:
        event_stream.subscribe(EventStreamSubscriber.MAIN, on_event, str(uuid4()))
    else:
        logger.warning(
            'Runtime does not have an event_stream attribute, cannot subscribe.'
        )

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
            fake_user_response_fn=cast(FakeUserResponseFunc, handle_awaiting_input),
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

            # Attempt to extract comments by searching backwards through history
            # NEW LOGIC: Attempt to extract comments from the AgentFinishAction message
            parse_error: str | None = None
            found_review_in_finish = False

            if agent_history:
                last_event = agent_history[-1]
                if isinstance(last_event, AgentFinishAction):
                    logger.info(
                        f'Agent finished. Attempting to parse review from final_thought: {last_event.final_thought[:200]}...'
                    )
                    logger.debug(
                        f'Full final_thought content: >>>{last_event.final_thought}<<<'
                    )  # DEBUG
                    try:
                        # Attempt to parse the final_thought directly as JSON
                        parsed_content = json.loads(last_event.final_thought.strip())
                        if isinstance(parsed_content, list):
                            # Found a list, try to validate it
                            validated_comments = []
                            for c_dict in parsed_content:
                                # === Start: Reused Validation Logic ===
                                if isinstance(c_dict, dict) and 'comment' in c_dict:
                                    path = c_dict.get('path')
                                    line = c_dict.get('line')
                                    comment_text = c_dict['comment']
                                    valid_comment = True

                                    if path is not None and not isinstance(path, str):
                                        logger.warning(
                                            f'Skipping comment with invalid path type: {c_dict}'
                                        )
                                        valid_comment = False
                                    if line is not None and not isinstance(line, int):
                                        if isinstance(line, str) and line.isdigit():
                                            line = int(line)
                                        else:
                                            logger.warning(
                                                f'Skipping comment with invalid line type: {c_dict}'
                                            )
                                            valid_comment = False
                                    if not isinstance(comment_text, str):
                                        logger.warning(
                                            f'Skipping comment with invalid comment text type: {c_dict}'
                                        )
                                        valid_comment = False

                                    if valid_comment:
                                        validated_comments.append(
                                            ReviewComment(
                                                path=path,
                                                comment=comment_text,
                                                line=line,
                                            )
                                        )
                                else:
                                    logger.warning(
                                        f'Skipping invalid comment structure: {c_dict}'
                                    )
                                # === End: Reused Validation Logic ===

                            if validated_comments:
                                comments = validated_comments
                                found_review_in_finish = True
                                logger.info(
                                    f'Extracted {len(comments)} review comments from AgentFinishAction final_thought.'
                                )
                            else:
                                # It was a list, but contained no valid comments
                                parse_error = 'Agent finish message was a list but contained no valid comment objects.'
                                logger.warning(
                                    f'{parse_error} Final thought snippet: {last_event.final_thought[:200]}'
                                )

                        else:
                            # Content was valid JSON, but not a list
                            parse_error = (
                                'Agent finish message content was not a JSON list.'
                            )
                            logger.warning(
                                f'{parse_error} Final thought snippet: {last_event.final_thought[:200]}'
                            )

                    except json.JSONDecodeError as e:
                        parse_error = (
                            f'Failed to parse agent finish message as JSON: {e}'
                        )
                        logger.warning(
                            f'{parse_error} Final thought snippet: {last_event.final_thought[:200]}'
                        )
                    except Exception as e:
                        parse_error = f'Error processing agent finish message: {e}'
                        logger.warning(
                            f'{parse_error} Final thought snippet: {last_event.final_thought[:200]}'
                        )
                else:
                    # Last event was not AgentFinishAction
                    error_message = f'Agent did not end with AgentFinishAction. Last event: {type(last_event).__name__}'
                    logger.error(error_message)
            else:
                # No history
                error_message = 'Agent produced no history.'
                logger.error(error_message)

            # Determine final success/error state
            if found_review_in_finish and final_agent_state == AgentState.FINISHED:
                success = True
                error_message = None  # Clear any previous agent loop error
                logger.info('Review successfully extracted from AgentFinishAction.')
            elif final_agent_state == AgentState.ERROR:
                success = False
                # Keep the original error_message from the agent loop if it exists
                if not error_message:
                    error_message = 'Agent finished in ERROR state.'
                logger.error(f'Agent finished in ERROR state: {error_message}')
            else:
                # Covers cases: No history, last event not Finish, Finish message invalid/empty, agent finished unexpectedly
                success = False
                if (
                    not error_message
                ):  # Only set if no specific error was already logged
                    if parse_error:
                        error_message = f'Failed to extract review from finish message: {parse_error}'
                    elif final_agent_state != AgentState.FINISHED:
                        error_message = f'Agent finished in unexpected state ({final_agent_state}) and no valid review found.'
                    else:  # Should imply finish state but parsing failed or last event wasn't finish
                        error_message = (
                            'Agent finished but review could not be extracted.'
                        )
                logger.error(f'Review processing failed: {error_message}')

    except Exception:
        # Catch any other unexpected errors during processing
        logger.exception('An unexpected exception occurred during agent execution:')
        success = False
        final_agent_state = AgentState.ERROR  # Assume error state

    finally:
        # Ensure runtime is closed if it was created
        if runtime:
            runtime.close()  # Sync close

    # Construct the final output
    output = ReviewerOutput(
        pr_info=pr_data,
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
        final_agent_state=final_agent_state,
    )

    return output


# Helper function for JSON serialization
def json_default(obj):
    if isinstance(obj, BaseModel):  # Handle Pydantic models (including Issue)
        return obj.model_dump()
    if dataclasses.is_dataclass(obj):
        # Handle other dataclasses
        return dataclasses.asdict(obj)
    if isinstance(obj, SecretStr):
        return obj.get_secret_value()  # Convert SecretStr to str
    # For other types, try converting to string as a fallback
    try:
        return str(obj)
    except Exception:
        raise TypeError(
            f'Object of type {obj.__class__.__name__} is not JSON serializable'
        )


def write_output_to_file(output_file: str, output_data: ReviewerOutput):
    """Writes the ReviewerOutput data to the specified JSONL file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(
                dataclasses.asdict(output_data), f, indent=2, default=json_default
            )
        logger.info(f'Successfully wrote output to {output_file}')
    except Exception as e:
        logger.error(f'Failed to write output to {output_file}: {e}')
        # Fallback: print to stdout if writing fails
        print(
            json.dumps(dataclasses.asdict(output_data), indent=2, default=json_default)
        )


async def run_review_task(
    pr_url: str,
    review_level: str,
    review_depth: str,
    token: str,
    username: str,
    max_iterations: int,
    output_dir: str,  # Keep output_dir for potential future use, though not used for printing
    output_file: str,
    llm_config: LLMConfig,
    base_container_image: str | None,
    runtime_container_image: str | None,
    prompt_file: str | None,
    repo_instruction_file: str | None,
    base_domain: str | None,
) -> None:
    """Orchestrates the code review process for a given PR URL."""
    logger.info(f'Starting review task for PR: {pr_url}')

    # 1. Identify platform and parse URL
    platform = await identify_token(token, base_domain)
    logger.info(f'Identified platform: {platform.value}')
    handler_class: type[IssueHandlerInterface]
    if platform == ProviderType.GITHUB:
        handler_class = GithubPRHandler
    elif platform == ProviderType.GITLAB:
        handler_class = GitlabPRHandler
    else:
        raise ValueError(f'Unsupported platform: {platform}')

    assert hasattr(
        handler_class, 'parse_pr_url'
    ), f'{handler_class.__name__} lacks parse_pr_url'
    owner, repo, issue_number = handler_class.parse_pr_url(pr_url)
    logger.info(f'Parsed PR URL: owner={owner}, repo={repo}, number={issue_number}')

    # 2. Create Issue Handler
    # Set default base_domain if None
    if base_domain is None:
        base_domain = 'github.com' if platform == ProviderType.GITHUB else 'gitlab.com'
    issue_handler: GithubPRHandler = handler_class(  # type: ignore[call-arg, assignment]
        owner=owner,
        repo=repo,
        token=token,
        username=username,
        base_domain=base_domain,  # Now guaranteed to be str
    )
    logger.info(f'Created issue handler: {type(issue_handler).__name__}')

    # 3. Fetch PR Info (Issue object)
    assert hasattr(
        issue_handler, 'get_converted_issues'
    ), f'{type(issue_handler).__name__} lacks get_converted_issues'

    try:
        # Fetch full PR details as a dictionary
        pr_data = await issue_handler.get_pr_details(issue_number)
        logger.info(f'Fetched PR data for #{pr_data["number"]}')
        logger.info(f'Type of pr_data: {type(pr_data)}')
        logger.info(
            f'Content of pr_data (keys): {list(pr_data.keys())}'
        )  # Log keys for brevity
    except Exception as e:
        logger.error(f'Failed to fetch PR info: {e}')
        # Print error output similar to main's exception handling
        error_output = ReviewerOutput(
            pr_info=Issue(number=issue_number, url=pr_url),  # Basic info
            review_level=review_level,
            review_depth=review_depth,
            instruction='',
            history=[],
            success=False,
            error=f'Failed to fetch PR info: {e}',
            final_agent_state=AgentState.ERROR,
        )
        write_output_to_file(output_file, error_output)
        return  # Exit early

    # 4. Setup repository directory
    repo_dir = os.path.join(output_dir, 'repo')  # Use output_dir for repo checkout
    os.makedirs(repo_dir, exist_ok=True)
    logger.info(f'Repository directory set to: {repo_dir}')

    # 5. Checkout PR branch
    try:
        assert hasattr(
            issue_handler, 'checkout_pr'
        ), f'{type(issue_handler).__name__} lacks checkout_pr'
        await issue_handler.checkout_pr(pr_data['number'], repo_dir)
        logger.info(f"Checked out PR branch for #{pr_data['number']} into {repo_dir}")
        # base_commit = await issue_handler.get_head_commit(repo_dir) # Not needed by process_review
        # logger.info(f'Base commit set to: {base_commit}')
    except Exception as e:
        logger.error(f'Failed to checkout PR branch: {e}')
        error_output = ReviewerOutput(
            pr_info=pr_data,
            review_level=review_level,
            review_depth=review_depth,
            instruction='',
            history=[],
            success=False,
            error=f'Failed to checkout PR branch: {e}',
            final_agent_state=AgentState.ERROR,
        )
        print(json.dumps(error_output, indent=2, default=json_default))
        return  # Exit early

    # 6. Read repository instructions if provided
    repo_instruction: str | None = None
    if repo_instruction_file:
        try:
            async with aiofiles.open(repo_instruction_file, mode='r') as f:
                repo_instruction = await f.read()
            logger.info(f'Read repository instructions from: {repo_instruction_file}')
        except Exception as e:
            logger.warning(
                f'Could not read repository instruction file {repo_instruction_file}: {e}'
            )
            # Continue without repo instructions if file reading fails

    # 7. Read prompt template
    if prompt_file is None:
        # Use default prompt if none provided
        prompt_file = os.path.join(
            os.path.dirname(__file__), 'prompts/review/basic-review.jinja'
        )
        logger.info(f'Using default prompt template: {prompt_file}')

    try:
        async with aiofiles.open(prompt_file, mode='r') as f:
            prompt_template = await f.read()
        logger.info(f'Read prompt template from: {prompt_file}')
    except Exception as e:
        logger.error(f'Failed to read prompt template file {prompt_file}: {e}')
        error_output = ReviewerOutput(
            pr_info=pr_data,
            review_level=review_level,
            review_depth=review_depth,
            instruction='',
            history=[],
            success=False,
            error=f'Failed to read prompt template: {e}',
            final_agent_state=AgentState.ERROR,
        )
        write_output_to_file(output_file, error_output)
        return  # Exit early

    # 9. Process the PR using the core logic function
    try:
        logger.info(f'Passing to process_review - Type of pr_data: {type(pr_data)}')
        logger.info(
            f'Passing to process_review - Content of pr_data (keys): {list(pr_data.keys())}'
        )
        output = await process_review(
            pr_data=pr_data,  # Pass the dictionary
            platform=platform,
            max_iterations=max_iterations,
            llm_config=llm_config,
            output_dir=output_dir,  # Pass output_dir for workspace creation inside process_review
            base_container_image=base_container_image,
            runtime_container_image=runtime_container_image,
            prompt_template=prompt_template,
            repo_dir=repo_dir,  # Pass the checkout location
            repo_instruction=repo_instruction,
            reset_logger=False,  # Assuming single process, no need to reset logger
            review_level=review_level,
            review_depth=review_depth,
        )

        # Check if the first attempt failed and might benefit from a retry with higher temperature
        # We retry if it wasn't successful AND the agent didn't finish cleanly (e.g., ERROR or RUNNING/INIT)
        # AgentState.STOPPED might indicate a deliberate stop, so we don't retry then.
        # AgentState.AWAITING_USER_INPUT should be handled by fake_user_response_fn, but check just in case.
        needs_retry = not output.success and output.final_agent_state in [
            AgentState.ERROR,
            AgentState.RUNNING,
            AgentState.LOADING,
            AgentState.AWAITING_USER_INPUT,
        ]

        if needs_retry:
            logger.warning(
                f'Initial review attempt failed or did not complete cleanly (State: {output.final_agent_state}). Retrying with temperature=2.0.'
            )
            # Create a new LLMConfig for the retry, inheriting settings but changing temperature
            retry_llm_config = dataclasses.replace(llm_config, temperature=2.0)

            # Call process_review again with the retry config
            output = await process_review(
                pr_data=pr_data,
                platform=platform,
                max_iterations=max_iterations,
                llm_config=retry_llm_config,  # Use retry config
                output_dir=output_dir,
                base_container_image=base_container_image,
                runtime_container_image=runtime_container_image,
                prompt_template=prompt_template,
                repo_dir=repo_dir,
                repo_instruction=repo_instruction,
                reset_logger=False,
                review_level=review_level,
                review_depth=review_depth,
            )

        # Write the final output (either from first attempt or retry) to file
        write_output_to_file(output_file, output)
        if output.success:
            logger.info('Review task completed successfully.')
        else:
            logger.warning(
                f'Review task finished with success=False. Final agent state: {output.final_agent_state}. Error: {output.error}'
            )

    except Exception as e:
        logger.error(f'An unexpected error occurred during review processing: {e}')
        # Create a generic error output if processing fails unexpectedly
        error_output = ReviewerOutput(
            pr_info=pr_data,
            review_level=review_level,
            review_depth=review_depth,
            instruction='',  # May not have been generated
            history=[],
            success=False,
            error=f'Review processing failed: {e}',
            final_agent_state=AgentState.ERROR,
        )
        write_output_to_file(output_file, error_output)


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
        '--output-file',
        type=str,
        required=True,
        help='Path to the output JSONL file.',
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
        '--llm-temperature',
        type=float,
        default=1.0,  # Default to 1.0 as before
        help='Temperature for the LLM',
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
        '--llm-num-retries',
        type=int,
        default=3,  # Default number of retries
        help='Number of retries for LLM API calls.',
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

    review_level = my_args.review_level  # noqa: F841
    review_depth = my_args.review_depth  # noqa: F841
    output_dir = my_args.output_dir  # noqa: F841
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
        num_retries=my_args.llm_num_retries,  # Use the argument here
        temperature=my_args.llm_temperature,  # Use the argument here
    )

    # Only set api_version if it was explicitly provided, otherwise let LLMConfig handle it
    if api_version is not None:
        llm_config.api_version = api_version

    # Set default prompt file if not provided
    prompt_file = my_args.prompt_file
    if prompt_file is None:
        # Use a default review prompt (adjust path as needed)
        prompt_file = os.path.join(
            os.path.dirname(__file__), 'prompts/review/basic-review.jinja'
        )
        logger.info(f'Prompt file not specified, using default: {prompt_file}')

    # Construct pr_url
    base_domain_val = my_args.base_domain
    if base_domain_val is None:
        base_domain_val = (
            'github.com' if platform == ProviderType.GITHUB else 'gitlab.com'
        )
    # Adjust URL format based on platform
    pr_number = my_args.pr_number  # Need pr_number here
    if platform == ProviderType.GITLAB:
        pr_url = (
            f'https://{base_domain_val}/{owner}/{repo}/-/merge_requests/{pr_number}'
        )
    else:  # Default to GitHub format
        pr_url = f'https://{base_domain_val}/{owner}/{repo}/pull/{pr_number}'
    logger.info(f'Constructed PR URL: {pr_url}')

    repo_instruction_file = my_args.repo_instruction_file  # Define file path variable
    asyncio.run(
        run_review_task(
            pr_url=pr_url,
            review_level=my_args.review_level,
            review_depth=my_args.review_depth,
            token=token,
            username=username,
            max_iterations=my_args.max_iterations,
            output_dir=my_args.output_dir,
            output_file=my_args.output_file,
            llm_config=llm_config,
            base_container_image=base_container_image,
            runtime_container_image=runtime_container_image,
            prompt_file=prompt_file,  # Pass file path
            repo_instruction_file=repo_instruction_file,  # Pass file path
            base_domain=my_args.base_domain,  # Pass original arg
        )
    )


if __name__ == '__main__':
    main()
