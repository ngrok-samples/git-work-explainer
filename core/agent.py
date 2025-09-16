"""
Main AI agent that orchestrates between git data sources and LLM APIs.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from pathlib import Path

from .models import (
    AnalysisRequest, 
    UserContext, 
    AgentResponse, 
    AudienceType, 
    RepositoryContext,
    CommitInfo
)
from .llm_client import LLMClient, get_available_llm_client
from git_analyzer import GitAnalyzer
from interactive_prompter import InteractivePrompter


class WorkExplainerAgent:
    """
    Main AI agent that orchestrates between git data and LLM APIs.
    
    This class is designed to be easily wrappable as an MCP server tool.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, prefer_provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the agent with an optional LLM client.
        
        Args:
            llm_client: Optional LLM client instance
            prefer_provider: Optional preference for 'openai' or 'anthropic'
            model: Optional specific model to use
        """
        self.llm_client = llm_client or get_available_llm_client(prefer_provider, model)
        if not self.llm_client:
            raise RuntimeError("No LLM client available. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
        
        self.git_analyzer: Optional[GitAnalyzer] = None
        self.prompter = InteractivePrompter()
    
    def set_repository(self, repo_path: str = '.') -> None:
        """Set the git repository to analyze."""
        self.git_analyzer = GitAnalyzer(repo_path)
    
    async def explain_work(
        self,
        commit_count: int = 5,
        audience: AudienceType = AudienceType.PRODUCT_MANAGER,
        repo_path: str = '.',
        interactive: bool = True,
        user_context: Optional[UserContext] = None
    ) -> AgentResponse:
        """
        Main method to explain recent development work.
        
        This method coordinates the entire process:
        1. Analyze git commits
        2. Gather user context (if interactive)
        3. Send data to LLM for analysis
        4. Return structured response
        """
        
        # Set up git analyzer if not already done
        if not self.git_analyzer or self.git_analyzer.repo.working_dir != Path(repo_path).absolute():
            self.set_repository(repo_path)
        
        # Step 1: Analyze git commits
        print("🔍 Analyzing git repository...")
        commits = self.git_analyzer.get_recent_commits(commit_count)
        repository_context = self.git_analyzer.get_repository_context()
        
        if not commits:
            raise RuntimeError("No commits found in the repository")
        
        print(f"   Found {len(commits)} commits to analyze")
        
        # Step 2: Gather user context
        if interactive and user_context is None:
            print("\n💬 Gathering context...")
            user_context = self.prompter.get_user_context_for_audience(audience, commits)
        elif user_context is None:
            user_context = UserContext(target_audience=audience)
        
        # Step 3: Prepare analysis request
        request = AnalysisRequest(
            commits=commits,
            repository_context=repository_context,
            user_context=user_context
        )
        
        # Step 4: Send to LLM for analysis
        print(f"\n🤖 Analyzing with AI ({self.llm_client.__class__.__name__})...")
        try:
            response = await self.llm_client.analyze_commits(request)
            print(f"   Analysis completed ({response.processing_time:.1f}s, {response.tokens_used} tokens)")
            return response
        except Exception as e:
            raise RuntimeError(f"LLM analysis failed: {str(e)}")
    
    def explain_work_sync(self, **kwargs) -> AgentResponse:
        """Synchronous wrapper for explain_work."""
        return asyncio.run(self.explain_work(**kwargs))
    
    # MCP Server Interface Methods
    # These methods provide a clean interface that can be easily wrapped as MCP tools
    
    def get_repository_summary(self, repo_path: str = '.') -> Dict[str, Any]:
        """Get basic repository information - suitable for MCP tool."""
        try:
            self.set_repository(repo_path)
            context = self.git_analyzer.get_repository_context()
            commits = self.git_analyzer.get_recent_commits(5)
            
            return {
                "repository": {
                    "name": context.name,
                    "branch": context.branch,
                    "language": context.language,
                    "framework": context.framework,
                    "project_type": context.project_type
                },
                "recent_activity": {
                    "commit_count": len(commits),
                    "latest_commit": {
                        "sha": commits[0].sha if commits else None,
                        "message": commits[0].short_message if commits else None,
                        "date": commits[0].date.isoformat() if commits else None
                    }
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_commits_for_audience(
        self,
        audience_type: str,
        commit_count: int = 5,
        repo_path: str = '.',
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze commits for a specific audience - suitable for MCP tool."""
        try:
            # Convert string audience to enum
            audience_map = {
                'product_manager': AudienceType.PRODUCT_MANAGER,
                'executive': AudienceType.EXECUTIVE,
                'engineering_leadership': AudienceType.ENGINEERING_LEADERSHIP,
                'marketing': AudienceType.MARKETING,
                'client_stakeholder': AudienceType.CLIENT_STAKEHOLDER,
                'technical_team': AudienceType.TECHNICAL_TEAM
            }
            
            audience = audience_map.get(audience_type.lower(), AudienceType.PRODUCT_MANAGER)
            
            # Create user context from provided context
            user_context = UserContext(target_audience=audience)
            if context:
                user_context.project_goal = context.get('project_goal')
                user_context.business_impact = context.get('business_impact') 
                user_context.technical_challenges = context.get('technical_challenges')
                user_context.next_steps = context.get('next_steps')
                user_context.additional_context = context.get('additional_context')
            
            # Run analysis
            response = self.explain_work_sync(
                commit_count=commit_count,
                audience=audience,
                repo_path=repo_path,
                interactive=False,
                user_context=user_context
            )
            
            # Return structured response
            summary = response.summary
            return {
                "summary": {
                    "title": summary.title,
                    "executive_summary": summary.executive_summary,
                    "technical_overview": summary.technical_overview,
                    "business_impact": summary.business_impact,
                    "key_changes": summary.key_changes,
                    "next_steps": summary.next_steps,
                    "work_categories": [cat.value for cat in summary.work_categories],
                    "audience": summary.audience.value
                },
                "metadata": {
                    "processing_time": response.processing_time,
                    "confidence_score": response.confidence_score,
                    "tokens_used": response.tokens_used,
                    "commit_count": len(response.summary.metadata.get('commits', []))
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_quick_summary(self, repo_path: str = '.', commit_count: int = 3) -> Dict[str, Any]:
        """Generate a quick summary - suitable for MCP tool."""
        try:
            return self.analyze_commits_for_audience(
                audience_type='product_manager',
                commit_count=commit_count,
                repo_path=repo_path,
                context=None
            )
        except Exception as e:
            return {"error": str(e)}


# Standalone functions for MCP server integration
def create_agent() -> WorkExplainerAgent:
    """Factory function to create a work explainer agent."""
    return WorkExplainerAgent()


def explain_git_work(
    repo_path: str = '.',
    audience: str = 'product_manager',
    commit_count: int = 5,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Standalone function to explain git work - perfect for MCP server wrapping.
    
    Args:
        repo_path: Path to git repository
        audience: Target audience type
        commit_count: Number of recent commits to analyze
        context: Additional context dictionary
    
    Returns:
        Dictionary with analysis results or error information
    """
    try:
        agent = create_agent()
        return agent.analyze_commits_for_audience(
            audience_type=audience,
            commit_count=commit_count,
            repo_path=repo_path,
            context=context
        )
    except Exception as e:
        return {"error": f"Failed to explain git work: {str(e)}"}


def get_repo_info(repo_path: str = '.') -> Dict[str, Any]:
    """Get repository information - perfect for MCP server wrapping."""
    try:
        agent = create_agent()
        return agent.get_repository_summary(repo_path)
    except Exception as e:
        return {"error": f"Failed to get repository info: {str(e)}"}
