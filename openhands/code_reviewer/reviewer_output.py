import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class ReviewComment:
    path: Optional[str] = None  # File path relative to repo root
    line: Optional[int] = None  # Line number for line-specific comments
    comment: str = ''  # The review comment text


@dataclasses.dataclass
class ReviewerOutput:
    pr_info: Dict[
        str, Any
    ]  # Using dict to store PR info (number, title, owner, repo, etc.)
    review_level: str  # e.g., 'line', 'file', 'pr'
    review_depth: str  # e.g., 'quick', 'deep'
    instruction: str  # The instruction given to the agent
    history: List[dict[str, Any]]  # Agent history (actions/observations)
    comments: List[ReviewComment] = dataclasses.field(
        default_factory=list
    )  # List of review comments
    metrics: Optional[dict[str, Any]] = None  # Agent metrics
    final_agent_state: Optional[str] = (
        None  # Final state of the agent (e.g., FINISHED, ERROR)
    )
    success: bool = False  # Whether the review process completed successfully
    error: Optional[str] = None  # Error message if success is False
