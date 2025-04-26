import argparse
import asyncio
import json
import os

from openhands.code_reviewer.reviewer_output import ReviewerOutput
from openhands.core.logger import openhands_logger as logger
from openhands.integrations.service_types import ProviderType
from openhands.resolver.interfaces.github import GithubPRHandler


def get_pr_handler(
    owner: str,
    repo: str,
    token: str | None,
    platform: ProviderType,
) -> GithubPRHandler:  # Return specific type now
    """Get the GitHub PR handler. Raises error for other platforms."""
    if platform != ProviderType.GITHUB:
        raise ValueError(
            f'Unsupported platform for code review comments: {platform}. Only GitHub is supported.'
        )

    gh_token = token or os.environ.get('GITHUB_TOKEN')
    if not gh_token:
        raise ValueError('GitHub token is required for GitHub PR handler')

    return GithubPRHandler(token=gh_token, owner=owner, repo=repo)


def post_comments(
    output_file: str,
    token: str | None,
    selected_repo: str,
    pr_number: int,
):
    from openhands.code_reviewer.reviewer_output import ReviewComment

    """Reads review output and posts comments to the PR."""
    logger.info(f'Reading review output from: {output_file}')
    try:
        with open(output_file, 'r') as f:
            # Read the entire file content
            file_content = f.read()
            if not file_content:
                logger.error(f'Output file is empty: {output_file}')
                return
            output_data = json.loads(file_content)
            # Manually construct ReviewComment objects
            comments_data = output_data.pop(
                'comments', []
            )  # Get comments list, remove from dict
            comments_objects = [ReviewComment(**c) for c in comments_data]
            # Construct ReviewerOutput, passing the objects list
            review_output = ReviewerOutput(**output_data, comments=comments_objects)
    except FileNotFoundError:
        logger.error(f'Output file not found: {output_file}')
        return
    except json.JSONDecodeError:
        logger.error(f'Failed to decode JSON from output file: {output_file}')
        return
    except Exception as e:
        logger.error(f'Error reading or parsing output file {output_file}: {e}')
        return

    if not review_output.success:
        logger.error(f'Review generation failed. Error: {review_output.error}')
        # Optionally post a general failure comment? For now, just log.
        return

    if not review_output.comments:
        logger.warning('Review was successful, but no comments were generated.')
        # Optionally post a comment indicating review completed with no findings?
        return

    logger.info(f'Successfully parsed {len(review_output.comments)} comments.')

    try:
        owner, repo = selected_repo.split('/')
    except ValueError:
        logger.error(f'Invalid repository format: {selected_repo}. Use owner/repo.')
        return

    # Assume GitHub platform
    platform = ProviderType.GITHUB
    try:
        pr_handler = get_pr_handler(owner, repo, token, platform)
    except ValueError as e:  # Catch specific error from get_pr_handler
        logger.error(f'Configuration error getting PR handler: {e}')
        return

    logger.info(
        f'Posting {len(review_output.comments)} comments to PR #{pr_number} on {platform.value}...'
    )

    # Post comments using the handler
    # The handler interface might need adjustment if post_review doesn't exist
    # or takes different arguments. Assuming a method like post_review(pr_number, comments)
    # Check if the handler has the post_review method
    if not hasattr(pr_handler, 'post_review'):
        logger.error(f'{type(pr_handler).__name__} does not have a post_review method.')
        return
    comments_to_post = review_output.comments
    try:
        asyncio.run(
            pr_handler.post_review(pr_number=pr_number, comments=comments_to_post)
        )
    except Exception:  # Catch errors during comment posting
        logger.exception(
            f'Failed to post comments to PR #{pr_number}'
        )  # Use logger.exception for stack trace
        return  # Exit if posting fails

    logger.info(f'Successfully posted comments to PR #{pr_number}.')


def main():
    parser = argparse.ArgumentParser(description='Post review comments to a PR.')
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to the review_output.jsonl file.',
    )
    parser.add_argument(
        '--selected-repo',
        type=str,
        required=True,
        help='Repository where the PR exists in the format `owner/repo`.',
    )
    parser.add_argument(
        '--pr-number',
        type=int,
        required=True,
        help='Pull Request number to post comments to.',
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Platform token (GitHub PAT or GitLab access token). Reads from env vars (GITHUB_TOKEN/GITLAB_TOKEN) if not provided.',
    )

    args = parser.parse_args()

    post_comments(
        output_file=args.output_file,
        token=args.token,
        selected_repo=args.selected_repo,
        pr_number=args.pr_number,
    )


if __name__ == '__main__':
    main()
