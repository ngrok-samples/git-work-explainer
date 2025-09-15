"""
LLM client interface and implementations.
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import asdict

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from core.models import AnalysisRequest, WorkSummary, AgentResponse, AudienceType, WorkType


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def analyze_commits(self, request: AnalysisRequest) -> AgentResponse:
        """Analyze commits and generate a summary."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this LLM client is available (API key set, etc.)."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client for commit analysis."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        if self.api_key:
            openai.api_key = self.api_key
    
    def is_available(self) -> bool:
        """Check if OpenAI client is properly configured."""
        return OPENAI_AVAILABLE and self.api_key is not None
    
    async def analyze_commits(self, request: AnalysisRequest) -> AgentResponse:
        """Analyze commits using OpenAI's API."""
        start_time = time.time()
        
        # Build the system prompt
        system_prompt = self._build_system_prompt(request.user_context.target_audience)
        
        # Build the user prompt with commit data
        user_prompt = self._build_analysis_prompt(request)
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            processing_time = time.time() - start_time
            
            # Parse the response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Extract structured summary from the response
            summary = self._parse_llm_response(content, request.user_context.target_audience)
            
            return AgentResponse(
                summary=summary,
                raw_analysis=content,
                confidence_score=0.85,  # Could be improved with response analysis
                processing_time=processing_time,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def _build_system_prompt(self, audience: AudienceType) -> str:
        """Build system prompt based on target audience."""
        audience_instructions = {
            AudienceType.PRODUCT_MANAGER: """
You are an expert technical communicator helping developers explain their work to product managers.
Focus on: user impact, feature delivery, business value, timeline implications, and risk mitigation.
Use business-friendly language while maintaining technical accuracy.
""",
            AudienceType.EXECUTIVE: """
You are an expert technical communicator helping developers explain their work to executives.
Focus on: business impact, strategic value, resource efficiency, risk management, and competitive advantages.
Use high-level, strategic language with minimal technical jargon.
""",
            AudienceType.ENGINEERING_LEADERSHIP: """
You are an expert technical communicator helping developers explain their work to engineering leadership.
Focus on: technical decisions, architecture implications, team productivity, technical debt, and scalability.
Use technical language appropriate for engineering leaders.
""",
            AudienceType.MARKETING: """
You are an expert technical communicator helping developers explain their work to marketing teams.
Focus on: user-facing features, customer benefits, competitive advantages, and market positioning.
Use customer-centric language that translates technical work into market value.
""",
            AudienceType.CLIENT_STAKEHOLDER: """
You are an expert technical communicator helping developers explain their work to client stakeholders.
Focus on: delivered value, progress toward goals, quality assurance, and future capabilities.
Use clear, professional language that builds confidence and trust.
""",
            AudienceType.TECHNICAL_TEAM: """
You are an expert technical communicator helping developers explain their work to technical team members.
Focus on: implementation details, technical decisions, code quality, and knowledge sharing.
Use precise technical language with implementation details.
"""
        }
        
        base_instructions = """
Analyze the provided git commits and generate a structured summary of the development work.

Your response should include:
1. A compelling title for the work
2. Executive summary (2-3 sentences)
3. Technical overview appropriate for the audience
4. Business impact and value
5. Key changes and accomplishments
6. Next steps or recommendations

Be concise, accurate, and focus on what matters most to the target audience.
Structure your response as JSON with the following format:

{
  "title": "Work Summary Title",
  "executive_summary": "Brief overview...",
  "technical_overview": "Technical details...",
  "business_impact": "Business value...",
  "key_changes": ["Change 1", "Change 2", "Change 3"],
  "next_steps": "Recommended next steps...",
  "work_categories": ["feature", "bug_fix", "refactor"]
}
"""
        
        return base_instructions + "\n\n" + audience_instructions.get(audience, "")
    
    def _build_analysis_prompt(self, request: AnalysisRequest) -> str:
        """Build the analysis prompt with commit data."""
        prompt_parts = []
        
        # Repository context
        repo = request.repository_context
        prompt_parts.append(f"""
REPOSITORY CONTEXT:
- Name: {repo.name}
- Branch: {repo.branch}
- Type: {repo.project_type or 'Unknown'}
- Primary Language: {repo.language or 'Unknown'}
- Framework: {repo.framework or 'Unknown'}
""")
        
        # User context
        user_ctx = request.user_context
        if user_ctx.project_goal:
            prompt_parts.append(f"PROJECT GOAL: {user_ctx.project_goal}")
        if user_ctx.business_impact:
            prompt_parts.append(f"BUSINESS IMPACT: {user_ctx.business_impact}")
        if user_ctx.technical_challenges:
            prompt_parts.append(f"TECHNICAL CHALLENGES: {user_ctx.technical_challenges}")
        if user_ctx.additional_context:
            prompt_parts.append(f"ADDITIONAL CONTEXT: {user_ctx.additional_context}")
        
        # Commit details
        prompt_parts.append("\nCOMMIT ANALYSIS:")
        prompt_parts.append(f"Analyzing {len(request.commits)} commits:")
        
        for i, commit in enumerate(request.commits, 1):
            files_summary = f"{len(commit.changed_files)} files changed"
            if commit.changed_files:
                file_types = set()
                for file_change in commit.changed_files[:5]:  # Show first 5 files
                    ext = file_change.path.split('.')[-1] if '.' in file_change.path else 'no-ext'
                    file_types.add(ext)
                files_summary += f" ({', '.join(sorted(file_types))})"
            
            prompt_parts.append(f"""
{i}. Commit {commit.sha}
   Date: {commit.date.strftime('%Y-%m-%d %H:%M')}
   Author: {commit.author}
   Message: {commit.message}
   Changes: {files_summary}
   Diff Summary: {commit.diff_summary[:200]}...
""")
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, content: str, audience: AudienceType) -> WorkSummary:
        """Parse LLM response into structured WorkSummary."""
        try:
            # Try to extract JSON from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
                
                # Map string work categories to enum
                work_categories = []
                for cat in data.get('work_categories', []):
                    try:
                        work_categories.append(WorkType(cat))
                    except ValueError:
                        pass  # Skip invalid categories
                
                return WorkSummary(
                    title=data.get('title', 'Development Work Summary'),
                    executive_summary=data.get('executive_summary', ''),
                    technical_overview=data.get('technical_overview', ''),
                    business_impact=data.get('business_impact', ''),
                    key_changes=data.get('key_changes', []),
                    next_steps=data.get('next_steps', ''),
                    work_categories=work_categories,
                    audience=audience
                )
            else:
                # Fallback: parse unstructured response
                return self._parse_unstructured_response(content, audience)
                
        except json.JSONDecodeError:
            return self._parse_unstructured_response(content, audience)
    
    def _parse_unstructured_response(self, content: str, audience: AudienceType) -> WorkSummary:
        """Fallback parsing for unstructured LLM responses."""
        lines = content.split('\n')
        
        return WorkSummary(
            title="Development Work Summary",
            executive_summary=content[:200] + "..." if len(content) > 200 else content,
            technical_overview="Technical analysis provided by LLM",
            business_impact="Business impact analysis in progress",
            key_changes=["Multiple changes analyzed"],
            next_steps="Next steps to be determined",
            work_categories=[WorkType.FEATURE],  # Default
            audience=audience
        )


class AnthropicClient(LLMClient):
    """Anthropic Claude client for commit analysis."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not available. Install with: pip install anthropic")
        
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def is_available(self) -> bool:
        """Check if Anthropic client is properly configured."""
        return ANTHROPIC_AVAILABLE and self.api_key is not None
    
    async def analyze_commits(self, request: AnalysisRequest) -> AgentResponse:
        """Analyze commits using Anthropic's API."""
        start_time = time.time()
        
        # Build the system prompt
        system_prompt = self._build_system_prompt(request.user_context.target_audience)
        
        # Build the user prompt with commit data
        user_prompt = self._build_analysis_prompt(request)
        
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            processing_time = time.time() - start_time
            
            # Parse the response
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            # Extract structured summary from the response
            summary = self._parse_llm_response(content, request.user_context.target_audience)
            
            return AgentResponse(
                summary=summary,
                raw_analysis=content,
                confidence_score=0.85,  # Could be improved with response analysis
                processing_time=processing_time,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")
    
    def _build_system_prompt(self, audience: AudienceType) -> str:
        """Build system prompt based on target audience (same as OpenAI)."""
        audience_instructions = {
            AudienceType.PRODUCT_MANAGER: """
You are an expert technical communicator helping developers explain their work to product managers.
Focus on: user impact, feature delivery, business value, timeline implications, and risk mitigation.
Use business-friendly language while maintaining technical accuracy.
""",
            AudienceType.EXECUTIVE: """
You are an expert technical communicator helping developers explain their work to executives.
Focus on: business impact, strategic value, resource efficiency, risk management, and competitive advantages.
Use high-level, strategic language with minimal technical jargon.
""",
            AudienceType.ENGINEERING_LEADERSHIP: """
You are an expert technical communicator helping developers explain their work to engineering leadership.
Focus on: technical decisions, architecture implications, team productivity, technical debt, and scalability.
Use technical language appropriate for engineering leaders.
""",
            AudienceType.MARKETING: """
You are an expert technical communicator helping developers explain their work to marketing teams.
Focus on: user-facing features, customer benefits, competitive advantages, and market positioning.
Use customer-centric language that translates technical work into market value.
""",
            AudienceType.CLIENT_STAKEHOLDER: """
You are an expert technical communicator helping developers explain their work to client stakeholders.
Focus on: delivered value, progress toward goals, quality assurance, and future capabilities.
Use clear, professional language that builds confidence and trust.
""",
            AudienceType.TECHNICAL_TEAM: """
You are an expert technical communicator helping developers explain their work to technical team members.
Focus on: implementation details, technical decisions, code quality, and knowledge sharing.
Use precise technical language with implementation details.
"""
        }
        
        base_instructions = """
Analyze the provided git commits and generate a structured summary of the development work.

Your response should include:
1. A compelling title for the work
2. Executive summary (2-3 sentences)
3. Technical overview appropriate for the audience
4. Business impact and value
5. Key changes and accomplishments
6. Next steps or recommendations

Be concise, accurate, and focus on what matters most to the target audience.
Structure your response as JSON with the following format:

{
  "title": "Work Summary Title",
  "executive_summary": "Brief overview...",
  "technical_overview": "Technical details...",
  "business_impact": "Business value...",
  "key_changes": ["Change 1", "Change 2", "Change 3"],
  "next_steps": "Recommended next steps...",
  "work_categories": ["feature", "bug_fix", "refactor"]
}
"""
        
        return base_instructions + "\n\n" + audience_instructions.get(audience, "")
    
    def _build_analysis_prompt(self, request: AnalysisRequest) -> str:
        """Build the analysis prompt with commit data (same as OpenAI)."""
        prompt_parts = []
        
        # Repository context
        repo = request.repository_context
        prompt_parts.append(f"""
REPOSITORY CONTEXT:
- Name: {repo.name}
- Branch: {repo.branch}
- Type: {repo.project_type or 'Unknown'}
- Primary Language: {repo.language or 'Unknown'}
- Framework: {repo.framework or 'Unknown'}
""")
        
        # User context
        user_ctx = request.user_context
        if user_ctx.project_goal:
            prompt_parts.append(f"PROJECT GOAL: {user_ctx.project_goal}")
        if user_ctx.business_impact:
            prompt_parts.append(f"BUSINESS IMPACT: {user_ctx.business_impact}")
        if user_ctx.technical_challenges:
            prompt_parts.append(f"TECHNICAL CHALLENGES: {user_ctx.technical_challenges}")
        if user_ctx.additional_context:
            prompt_parts.append(f"ADDITIONAL CONTEXT: {user_ctx.additional_context}")
        
        # Commit details
        prompt_parts.append("\nCOMMIT ANALYSIS:")
        prompt_parts.append(f"Analyzing {len(request.commits)} commits:")
        
        for i, commit in enumerate(request.commits, 1):
            files_summary = f"{len(commit.changed_files)} files changed"
            if commit.changed_files:
                file_types = set()
                for file_change in commit.changed_files[:5]:  # Show first 5 files
                    ext = file_change.path.split('.')[-1] if '.' in file_change.path else 'no-ext'
                    file_types.add(ext)
                files_summary += f" ({', '.join(sorted(file_types))})"
            
            prompt_parts.append(f"""
{i}. Commit {commit.sha}
   Date: {commit.date.strftime('%Y-%m-%d %H:%M')}
   Author: {commit.author}
   Message: {commit.message}
   Changes: {files_summary}
   Diff Summary: {commit.diff_summary[:200]}...
""")
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, content: str, audience: AudienceType) -> WorkSummary:
        """Parse LLM response into structured WorkSummary (same as OpenAI)."""
        try:
            # Try to extract JSON from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
                
                # Map string work categories to enum
                work_categories = []
                for cat in data.get('work_categories', []):
                    try:
                        work_categories.append(WorkType(cat))
                    except ValueError:
                        pass  # Skip invalid categories
                
                return WorkSummary(
                    title=data.get('title', 'Development Work Summary'),
                    executive_summary=data.get('executive_summary', ''),
                    technical_overview=data.get('technical_overview', ''),
                    business_impact=data.get('business_impact', ''),
                    key_changes=data.get('key_changes', []),
                    next_steps=data.get('next_steps', ''),
                    work_categories=work_categories,
                    audience=audience
                )
            else:
                # Fallback: parse unstructured response
                return self._parse_unstructured_response(content, audience)
                
        except json.JSONDecodeError:
            return self._parse_unstructured_response(content, audience)
    
    def _parse_unstructured_response(self, content: str, audience: AudienceType) -> WorkSummary:
        """Fallback parsing for unstructured LLM responses (same as OpenAI)."""
        lines = content.split('\n')
        
        return WorkSummary(
            title="Development Work Summary",
            executive_summary=content[:200] + "..." if len(content) > 200 else content,
            technical_overview="Technical analysis provided by LLM",
            business_impact="Business impact analysis in progress",
            key_changes=["Multiple changes analyzed"],
            next_steps="Next steps to be determined",
            work_categories=[WorkType.FEATURE],  # Default
            audience=audience
        )


def get_available_llm_client(prefer_provider: Optional[str] = None) -> Optional[LLMClient]:
    """
    Get the first available LLM client.
    
    Args:
        prefer_provider: Optional preference for 'openai' or 'anthropic'
    """
    providers = []
    
    if prefer_provider == 'anthropic':
        providers = ['anthropic', 'openai']
    elif prefer_provider == 'openai':
        providers = ['openai', 'anthropic'] 
    else:
        # Default: try OpenAI first (for backward compatibility)
        providers = ['openai', 'anthropic']
    
    for provider in providers:
        try:
            if provider == 'openai':
                client = OpenAIClient()
                if client.is_available():
                    return client
            elif provider == 'anthropic':
                client = AnthropicClient()
                if client.is_available():
                    return client
        except ImportError:
            continue
    
    return None
