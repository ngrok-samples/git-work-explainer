"""
Data models for the git work explainer agent.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class AudienceType(Enum):
    """Different types of audiences for summaries."""
    PRODUCT_MANAGER = "product_manager"
    ENGINEERING_LEADERSHIP = "engineering_leadership"
    EXECUTIVE = "executive"
    MARKETING = "marketing"
    CLIENT_STAKEHOLDER = "client_stakeholder"
    TECHNICAL_TEAM = "technical_team"


class WorkType(Enum):
    """Categories of development work."""
    FEATURE = "feature"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    MAINTENANCE = "maintenance"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class FileChange:
    """Represents a file change in a commit."""
    path: str
    change_type: str  # added, modified, deleted, renamed
    additions: int = 0
    deletions: int = 0
    
    
@dataclass
class CommitInfo:
    """Rich commit information for LLM analysis."""
    sha: str
    full_sha: str
    message: str
    author: str
    author_email: str
    date: datetime
    changed_files: List[FileChange]
    diff_summary: str  # Brief diff summary
    
    @property
    def short_message(self) -> str:
        """Get the first line of commit message."""
        return self.message.split('\n')[0]


@dataclass
class RepositoryContext:
    """Repository context information."""
    name: str
    branch: str
    remote_url: Optional[str]
    language: Optional[str]  # Primary language
    framework: Optional[str]  # Detected framework
    project_type: Optional[str]  # web app, library, CLI tool, etc.


@dataclass
class UserContext:
    """Context provided by the user through interactive prompts."""
    project_goal: Optional[str] = None
    business_impact: Optional[str] = None
    technical_challenges: Optional[str] = None
    next_steps: Optional[str] = None
    urgency_level: Optional[str] = None
    additional_context: Optional[str] = None
    target_audience: AudienceType = AudienceType.PRODUCT_MANAGER


@dataclass
class AnalysisRequest:
    """Request object for LLM analysis."""
    commits: List[CommitInfo]
    repository_context: RepositoryContext
    user_context: UserContext
    analysis_type: str = "comprehensive"  # comprehensive, quick, technical


@dataclass
class WorkSummary:
    """Structured summary of development work."""
    title: str
    executive_summary: str
    technical_overview: str
    business_impact: str
    key_changes: List[str]
    next_steps: str
    work_categories: List[WorkType]
    audience: AudienceType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from the AI agent."""
    summary: WorkSummary
    raw_analysis: str
    confidence_score: float
    processing_time: float
    tokens_used: int = 0



