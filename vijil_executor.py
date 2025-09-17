#!/usr/bin/env python3
"""
Vijil Local Agent Executor for AI Git Work Explainer

This module provides integration with Vijil for evaluating the AI agent locally.
It creates adapters to make the agent compatible with Vijil's evaluation APIs.
"""

import os
import asyncio
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

# Vijil imports
try:
    from vijil import Vijil
    from vijil.local_agents.models import (
        ChatCompletionRequest,
        ChatCompletionResponse, 
        ChatCompletionChoice,
        ChatMessage,
    )
    from vijil.local_agents.constants import TERMINAL_STATUSES
    VIJIL_AVAILABLE = True
except ImportError:
    VIJIL_AVAILABLE = False
    print("Warning: Vijil SDK not installed. Please install with: pip install vijil")

# Local agent imports
from core.agent import WorkExplainerAgent
from core.models import AudienceType


def check_ngrok_auth():
    """Check if NGROK_AUTHTOKEN is set up properly."""
    if not os.getenv("NGROK_AUTHTOKEN"):
        print("⚠️  NGROK_AUTHTOKEN is not set!")
        print("   Please set your ngrok auth token in your environment variables:")
        print("   export NGROK_AUTHTOKEN=your_auth_token_here")
        print("\n   You can get your auth token from: https://dashboard.ngrok.com/get-started/your-authtoken")
        return False
    return True


def check_vijil_api_key():
    """Check if VIJIL_API_KEY is set up properly.""" 
    if not os.getenv("VIJIL_API_KEY"):
        print("⚠️  VIJIL_API_KEY is not set!")
        print("   Please set your Vijil API key in your environment variables:")
        print("   export VIJIL_API_KEY=your_api_key_here")
        return False
    return True


def get_available_harnesses():
    """Get list of available harnesses from Vijil (if possible)."""
    # Based on user testing and Vijil documentation, these are known working harnesses
    known_harnesses = [
        "security",
        "ethics", 
        "trust_score",
        # Add more as we discover what works
    ]
    
    try:
        if not VIJIL_AVAILABLE:
            return ["security"]  # Default fallback when SDK not available
            
        # Return the known working harnesses regardless of API key status
        # This allows users to see what's available even before configuring keys
        return known_harnesses
    except Exception:
        # Fallback to basic harness
        return ["security"]


class GitWorkExplainerExecutor:
    """
    Local Agent Executor for the Git Work Explainer agent.
    
    This class adapts the WorkExplainerAgent to work with Vijil's evaluation framework.
    """
    
    def __init__(self, repo_path: str = '.', llm_provider: Optional[str] = None):
        """
        Initialize the executor.
        
        Args:
            repo_path: Path to the git repository to analyze
            llm_provider: Optional LLM provider preference ('openai' or 'anthropic')
        """
        self.repo_path = Path(repo_path).absolute()
        self.llm_provider = llm_provider
        self.agent = None
        self._init_agent()
    
    def _init_agent(self):
        """Initialize the WorkExplainerAgent."""
        try:
            self.agent = WorkExplainerAgent(prefer_provider=self.llm_provider)
            # Test that the agent can access LLM client
            if hasattr(self.agent, 'llm_client') and self.agent.llm_client:
                provider_name = self.agent.llm_client.__class__.__name__
                if self.llm_provider:
                    print(f"✅ Agent initialized with {provider_name} (requested: {self.llm_provider})")
                else:
                    print(f"✅ Agent initialized with {provider_name}")
            else:
                print(f"⚠️  Agent initialized but no LLM client detected")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            raise
    
    def input_adapter(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        Convert Vijil's ChatCompletionRequest to agent input format.
        
        Args:
            request: Vijil's standard chat completion request
            
        Returns:
            Dictionary with agent parameters
        """
        try:
            print(f"🔍 Input adapter received request: {request}")
            
            # Extract the user's message content
            if not request.messages:
                raise ValueError("No messages in request")
            
            # Get the last user message
            user_message = None
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if not user_message:
                user_message = "Analyze recent development work for product managers"
            
            print(f"📝 Extracted user message: {user_message}")
            
            # Classify the question to determine how to respond
            question_type = self._classify_question(user_message)
            print(f"🔍 Question classified as: {question_type}")
            
            if question_type == 'git_related':
                # Parse the message to extract parameters for git analysis
                params = self._parse_user_message(user_message)
                
                # Create user context that includes the original question
                from core.models import UserContext
                user_context = UserContext(
                    additional_context=f"User request: {user_message}",
                    target_audience=params.get("audience", AudienceType.PRODUCT_MANAGER)
                )
                
                result = {
                    "commit_count": params.get("commit_count", 5),
                    "audience": params.get("audience", AudienceType.PRODUCT_MANAGER),
                    "repo_path": str(self.repo_path),
                    "interactive": False,  # Always non-interactive for Vijil
                    "user_context": user_context,
                    "question_type": "git_related"
                }
            else:
                # Out-of-scope question - prepare for refusal response
                result = {
                    "question_type": "out_of_scope",
                    "original_question": user_message,
                    "response_type": "decline"
                }
            
            print(f"🎯 Input adapter returning: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Input adapter error: {e}")
            raise
    
    def _parse_user_message(self, message: str) -> Dict[str, Any]:
        """
        Parse user message to extract agent parameters.
        
        This method tries to understand natural language requests and convert
        them to appropriate agent parameters.
        """
        message_lower = message.lower()
        params = {}
        
        # Parse commit count
        import re
        commit_match = re.search(r'(\d+)\s+commits?', message_lower)
        if commit_match:
            params["commit_count"] = int(commit_match.group(1))
        
        # Parse audience type
        audience_mapping = {
            'product manager': AudienceType.PRODUCT_MANAGER,
            'pm': AudienceType.PRODUCT_MANAGER,
            'executive': AudienceType.EXECUTIVE,
            'exec': AudienceType.EXECUTIVE,
            'engineering': AudienceType.ENGINEERING_LEADERSHIP,
            'engineering leadership': AudienceType.ENGINEERING_LEADERSHIP,
            'marketing': AudienceType.MARKETING,
            'client': AudienceType.CLIENT_STAKEHOLDER,
            'stakeholder': AudienceType.CLIENT_STAKEHOLDER,
            'technical': AudienceType.TECHNICAL_TEAM,
            'technical team': AudienceType.TECHNICAL_TEAM,
        }
        
        for keyword, audience in audience_mapping.items():
            if keyword in message_lower:
                params["audience"] = audience
                break
        
        return params
    
    def _classify_question(self, message: str) -> str:
        """
        Classify whether a question is git-related or out-of-scope.
        
        Args:
            message: The user's question/message
            
        Returns:
            'git_related' or 'out_of_scope'
        """
        message_lower = message.lower().strip()
        
        # Check for obvious out-of-scope patterns FIRST (higher priority)
        out_of_scope_patterns = [
            'capital of', 'what is the capital', 'who is', 'where is', 'when was',
            'geography', 'history', 'weather', 'recipe', 'cooking', 'france', 'paris',
            'country', 'city', 'poem', 'story', 'write a', 'tell me about',
            'overcome by', 'gloom', 'misery', 'death', 'guilty', 'penalty',
            'ignore previous', 'ignore all', 'forget', 'disregard'
        ]
        
        for pattern in out_of_scope_patterns:
            if pattern in message_lower:
                return 'out_of_scope'
        
        # Git/development related keywords (checked after out-of-scope)
        git_keywords = [
            'git', 'commit', 'commits', 'repository', 'repo', 'code', 'development', 
            'work', 'changes', 'analyze', 'analysis', 'project', 'feature', 'bug',
            'pull request', 'merge', 'branch', 'developer', 'programming', 'software',
            'technical', 'engineering', 'codebase', 'implementation', 'recent',
            'last', 'summary', 'explain', 'overview', 'what happened', 'progress'
        ]
        
        # Check for git-related keywords
        for keyword in git_keywords:
            if keyword in message_lower:
                return 'git_related'
        
        # For ambiguous cases, if the message is very short or asks general questions,
        # treat as potentially git-related to be safe
        if len(message_lower.split()) <= 3:
            return 'git_related'
        
        # Default to out-of-scope for safety (better to decline than hallucinate)
        return 'out_of_scope'
    
    def _create_refusal_response(self, original_question: str) -> Dict[str, Any]:
        """
        Create a professional refusal response for out-of-scope questions.
        
        Args:
            original_question: The original question that was out-of-scope
            
        Returns:
            Dictionary containing a polite refusal response
        """
        refusal_message = (
            f"I specialize in analyzing git repository development work and cannot answer "
            f"questions outside that scope. "
            f"\n\nI can help you with:\n"
            f"• Analyzing recent commits and code changes\n"
            f"• Summarizing development work for different audiences\n"
            f"• Explaining technical progress and business impact\n"
            f"• Reviewing project development history\n"
            f"\nPlease ask me about code commits, development work, or repository analysis."
        )
        
        return {
            'summary': {
                'title': 'Request Outside Agent Scope',
                'executive_summary': refusal_message,
                'technical_overview': 'This agent is designed specifically for git repository analysis.',
                'business_impact': 'N/A - Request is outside agent capabilities',
                'key_changes': ['Agent specializes in git analysis only'],
                'next_steps': 'Please ask questions related to git commits or development work.',
                'work_categories': [],
                'audience': 'general'
            },
            'response_type': 'scope_refusal',
            'original_question': original_question
        }
    
    def output_adapter(self, agent_output: Any) -> ChatCompletionResponse:
        """
        Convert agent output to Vijil's ChatCompletionResponse format.
        
        Args:
            agent_output: Response from the WorkExplainerAgent
            
        Returns:
            ChatCompletionResponse compatible with Vijil
        """
        try:
            print(f"🔍 Output adapter received: {type(agent_output)} - {agent_output}")
            
            # Format the agent response as a readable message
            if hasattr(agent_output, 'summary'):
                # Full agent response
                summary = agent_output.summary
                content = self._format_summary_as_text(summary)
                print(f"📝 Formatted full agent response")
            elif isinstance(agent_output, dict) and 'summary' in agent_output:
                # Dictionary response
                content = self._format_dict_summary(agent_output['summary'])
                print(f"📝 Formatted dict summary response")
            else:
                # Fallback to string representation
                content = str(agent_output)
                print(f"📝 Using string fallback: {content[:100]}...")
            
            # Create the response message
            response_message = ChatMessage(
                role="assistant",
                content=content,
                tool_calls=None,
                retrieval_context=None
            )
            
            # Create the choice
            choice = ChatCompletionChoice(
                index=0,
                message=response_message,
                finish_reason="stop"
            )
            
            # Create the full response
            response = ChatCompletionResponse(
                model="git-work-explainer",
                choices=[choice],
                usage=None,
            )
            
            print(f"🎯 Output adapter returning ChatCompletionResponse")
            return response
            
        except Exception as e:
            print(f"❌ Output adapter error: {e}")
            # Error handling - return error message
            error_message = ChatMessage(
                role="assistant",
                content=f"Error analyzing git work: {str(e)}",
                tool_calls=None,
                retrieval_context=None
            )
            
            choice = ChatCompletionChoice(
                index=0,
                message=error_message,
                finish_reason="stop"
            )
            
            return ChatCompletionResponse(
                model="git-work-explainer",
                choices=[choice],
                usage=None,
            )
    
    def _format_summary_as_text(self, summary) -> str:
        """Format an AgentSummary object as readable text."""
        return f"""# {summary.title}

**Target Audience:** {summary.audience.value.replace('_', ' ').title()}

## Executive Summary
{summary.executive_summary}

## Technical Overview  
{summary.technical_overview}

## Business Impact
{summary.business_impact}

## Key Changes
{chr(10).join(f'- {change}' for change in summary.key_changes)}

## Next Steps
{summary.next_steps}

**Work Categories:** {', '.join(cat.value.replace('_', ' ').title() for cat in summary.work_categories)}
"""
    
    def _format_dict_summary(self, summary_dict: Dict[str, Any]) -> str:
        """Format a dictionary summary as readable text."""
        return f"""# {summary_dict.get('title', 'Development Work Summary')}

**Target Audience:** {summary_dict.get('audience', 'Unknown').replace('_', ' ').title()}

## Executive Summary
{summary_dict.get('executive_summary', 'Not available')}

## Technical Overview
{summary_dict.get('technical_overview', 'Not available')}

## Business Impact
{summary_dict.get('business_impact', 'Not available')}

## Key Changes
{chr(10).join(f'- {change}' for change in summary_dict.get('key_changes', []))}

## Next Steps
{summary_dict.get('next_steps', 'Not available')}

**Work Categories:** {', '.join(summary_dict.get('work_categories', []))}
"""
    
    async def agent_function(self, adapted_input) -> Any:
        """
        Main agent function that Vijil will call.
        
        This is the core function that Vijil's LocalAgentExecutor will invoke.
        Args:
            adapted_input: The output from input_adapter, containing agent parameters
        """
        try:
            print(f"🤖 Agent function called with input type: {adapted_input.get('question_type', 'unknown')}")
            
            # Handle out-of-scope questions with polite refusal
            if isinstance(adapted_input, dict) and adapted_input.get('question_type') == 'out_of_scope':
                original_question = adapted_input.get('original_question', 'Unknown question')
                print(f"⚠️  Declining out-of-scope question: {original_question[:50]}...")
                return self._create_refusal_response(original_question)
            
            # Handle git-related questions with normal analysis
            elif isinstance(adapted_input, dict) and adapted_input.get('question_type') == 'git_related':
                print(f"✅ Processing git-related request")
                # Remove our custom fields before calling the agent
                clean_input = {k: v for k, v in adapted_input.items() if k not in ['question_type']}
                return await self.agent.explain_work(**clean_input)
            
            # Fallback for legacy behavior (shouldn't happen with new input adapter)
            elif isinstance(adapted_input, dict):
                print(f"⚠️  Legacy input format detected, processing as git analysis")
                return await self.agent.explain_work(**adapted_input)
            else:
                print(f"⚠️  Unexpected input format, trying direct call")
                return await self.agent.explain_work(adapted_input)
                
        except Exception as e:
            print(f"❌ Agent function error: {e}")
            print(f"   Adapted input was: {adapted_input}")
            raise


class VijilEvaluator:
    """
    Main class for running Vijil evaluations on the Git Work Explainer agent.
    """
    
    def __init__(self, repo_path: str = '.', llm_provider: Optional[str] = None, skip_agent_init: bool = False):
        """
        Initialize the Vijil evaluator.
        
        Args:
            repo_path: Path to the git repository to analyze
            llm_provider: Optional LLM provider preference ('openai' or 'anthropic')
            skip_agent_init: If True, skip agent initialization (useful for report generation only)
        """
        if not VIJIL_AVAILABLE:
            raise ImportError("Vijil SDK is not installed. Please install with: pip install vijil")
        
        self.repo_path = repo_path
        self.llm_provider = llm_provider
        self.skip_agent_init = skip_agent_init
        self.executor = None
        self.vijil = None
        self.local_agent = None
        
        # Only initialize executor if we need the agent
        if not skip_agent_init:
            self.executor = GitWorkExplainerExecutor(repo_path, llm_provider)
            # Check prerequisites
            self._check_setup()
    
    def _check_setup(self):
        """Check if all required environment variables and dependencies are set up."""
        print("🔍 Checking Vijil setup...")
        
        if not check_vijil_api_key():
            raise ValueError("VIJIL_API_KEY is required")
        
        if not check_ngrok_auth():
            raise ValueError("NGROK_AUTHTOKEN is required for local agent evaluation")
        
        print("✅ Environment variables configured")
        
        # Test the API key by trying a simple API call
        print("🔍 Testing Vijil API key authentication...")
        self._test_vijil_api_key()
    
    def _test_vijil_api_key(self):
        """Test if the Vijil API key is working."""
        try:
            if not self.vijil:
                self._init_vijil()
            
            # Try a simple API call to test authentication
            # Use the evaluations.list method which should work if API key is valid
            evaluations = self.vijil.evaluations.list(limit=1)
            print("✅ Vijil API key is working")
            print(f"   API response received (found {len(evaluations) if evaluations else 0} evaluations)")
            
        except Exception as e:
            error_msg = str(e)
            if "token validation failed" in error_msg.lower() or "unauthorized" in error_msg.lower():
                print("❌ VIJIL_API_KEY authentication failed")
                print("   Your API key might be:")
                print("   - Invalid or expired")
                print("   - Missing required permissions")
                print("   - Incorrectly formatted")
                print(f"   Current key starts with: {os.getenv('VIJIL_API_KEY', 'NOT_SET')[:8]}...")
                print("\n   💡 Please check your API key in the Vijil dashboard")
            else:
                print(f"❌ API connection test failed: {e}")
            raise
    
    async def test_agent_function(self):
        """Test the agent function with sample input."""
        print("🧪 Testing agent function...")
        try:
            test_input = {
                "commit_count": 3,
                "audience": AudienceType.PRODUCT_MANAGER,
                "repo_path": str(self.executor.repo_path),
                "interactive": False,
                "user_context": None
            }
            
            result = await self.executor.agent_function(test_input)
            print(f"✅ Agent function test successful: {type(result)}")
            return True
        except Exception as e:
            print(f"❌ Agent function test failed: {e}")
            return False
    
    def _init_vijil(self):
        """Initialize Vijil client and local agent executor."""
        try:
            # Initialize Vijil client
            self.vijil = Vijil(api_key=os.getenv("VIJIL_API_KEY"))
            print("✅ Vijil client initialized")
            
            # Only create local agent if we have an executor
            if self.executor:
                # Create local agent executor
                self.local_agent = self.vijil.local_agents.create(
                    agent_function=self.executor.agent_function,
                    input_adapter=self.executor.input_adapter,
                    output_adapter=self.executor.output_adapter,
                )
                print("✅ Local agent executor created")
            
        except Exception as e:
            print(f"❌ Failed to initialize Vijil: {e}")
            raise
    
    def run_evaluation(
        self,
        agent_name: str = "Git Work Explainer", 
        evaluation_name: str = "Local Agent Evaluation",
        harnesses: List[str] = None,
        rate_limit: int = 30,
        rate_limit_interval: int = 1
    ) -> None:
        """
        Run a simple evaluation using vijil.local_agents.evaluate().
        
        Args:
            agent_name: Name for the agent in evaluation
            evaluation_name: Name for the evaluation run
            harnesses: List of harnesses to run (e.g., ["security_Small", "ethics_Small"])
            rate_limit: Maximum requests per interval
            rate_limit_interval: Rate limit interval in minutes
        """
        if not self.vijil:
            self._init_vijil()
        
        if harnesses is None:
            harnesses = ["security"]  # Default to security evaluation
        
        print(f"\n🚀 Starting Vijil evaluation...")
        print(f"   Agent: {agent_name}")
        print(f"   Evaluation: {evaluation_name}")
        print(f"   Harnesses: {', '.join(harnesses)}")
        print(f"   Repository: {self.repo_path}")
        
        try:
            # Run evaluation
            self.vijil.local_agents.evaluate(
                agent_name=agent_name,
                evaluation_name=evaluation_name,
                agent=self.local_agent,
                harnesses=harnesses,
                rate_limit=rate_limit,
                rate_limit_interval=rate_limit_interval,
            )
            print("✅ Evaluation completed successfully!")
            
        except KeyboardInterrupt:
            print("\n⚠️  Evaluation cancelled by user")
        except Exception as e:
            error_msg = str(e)
            if "harnesses" in error_msg.lower() or "tag group" in error_msg.lower():
                print(f"❌ Invalid harness names: {harnesses}")
                print("   Try using standard harnesses like:")
                print("   - security (security evaluation)")
                print("   - ethics (ethics evaluation)")  
                print("   - trust_score (general trustworthiness)")
                print("   - privacy (privacy evaluation)")  
                print("\n   💡 Use --list-harnesses to see all available options")
                print(f"\n   Original error: {e}")
            elif "unauthorized" in error_msg.lower() or "token validation failed" in error_msg.lower():
                print(f"❌ Authentication failed: {e}")
                print("   This might be due to:")
                print("   - Invalid VIJIL_API_KEY")
                print("   - API key doesn't have necessary permissions")
                print("   - Temporary API issue")
                print("\n   💡 Try checking your API key or using --advanced for more detailed error handling")
            elif "api-key" in error_msg.lower() or "does not exist" in error_msg.lower():
                print(f"❌ API key management error: {e}")
                print("   The system is trying to use an auto-generated API key that doesn't exist.")
                print("   💡 Try using --advanced for manual key management or contact support")
            else:
                print(f"❌ Evaluation failed: {e}")
            raise
    
    def run_advanced_evaluation(
        self,
        agent_name: str = "Git Work Explainer",
        evaluation_name: str = "Advanced Local Agent Evaluation", 
        harnesses: List[str] = None,
        rate_limit: int = 30,
        rate_limit_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Run an advanced evaluation with manual registration and monitoring.
        
        This method provides more control over the evaluation process.
        
        Args:
            agent_name: Name for the agent in evaluation
            evaluation_name: Name for the evaluation run
            harnesses: List of harnesses to run
            rate_limit: Maximum requests per interval
            rate_limit_interval: Rate limit interval in minutes
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.vijil:
            self._init_vijil()
        
        if harnesses is None:
            harnesses = ["security"]  # Default harness
        
        print(f"\n🚀 Starting advanced Vijil evaluation...")
        print(f"   Agent: {agent_name}")
        print(f"   Evaluation: {evaluation_name}")
        print(f"   Harnesses: {', '.join(harnesses)}")
        
        server = None
        api_key_name = None
        
        try:
            # Step 1: Register the agent
            print("📝 Registering agent...")
            clean_agent_name = agent_name.lower().replace(" ", "-")
            print(f"   Using agent name: {clean_agent_name}")
            
            server, api_key_name = self.vijil.local_agents.register(
                agent_name=clean_agent_name,
                evaluator=self.local_agent,
                rate_limit=rate_limit,
                rate_limit_interval=rate_limit_interval,
            )
            print(f"✅ Agent registered at: {server.url}")
            print(f"   Generated API key name: {api_key_name}")
            
            # Step 2: Create and trigger evaluation
            print("🔄 Creating evaluation...")
            evaluation = self.vijil.evaluations.create(
                model_hub="custom",
                model_name="local-agent", 
                name=evaluation_name,
                api_key_name=api_key_name,
                model_url=f"{server.url}/v1",
                harnesses=harnesses,
            )
            
            evaluation_id = evaluation.get("id")
            print(f"✅ Evaluation {evaluation_id} started")
            
            # Step 3: Monitor progress
            print("⏳ Monitoring evaluation progress...")
            import time
            
            while True:
                try:
                    status_data = self.vijil.evaluations.get_status(evaluation_id)
                    status = status_data.get("status")
                    print(f"   Status: {status}")
                    
                    if status in TERMINAL_STATUSES:
                        print(f"✅ Evaluation completed with status: {status}")
                        break
                        
                    time.sleep(5)  # Wait 5 seconds before checking again
                    
                except KeyboardInterrupt:
                    print("\n⚠️  Monitoring cancelled by user")
                    break
                except Exception as e:
                    print(f"❌ Error checking status: {e}")
                    break
            
            return {
                "evaluation_id": evaluation_id,
                "status": status,
                "server_url": server.url,
                "results": status_data
            }
            
        except Exception as e:
            print(f"❌ Advanced evaluation failed: {e}")
            raise
        
        finally:
            # Step 4: Cleanup
            if server and api_key_name:
                try:
                    print("🧹 Cleaning up...")
                    self.vijil.agents.deregister(server, api_key_name)
                    print("✅ Agent deregistered successfully")
                except Exception as cleanup_error:
                    print(f"⚠️  Cleanup warning: {cleanup_error}")
    
    def run_manual_evaluation(
        self,
        agent_name: str,
        evaluation_name: str,
        api_key_name: str,
        harnesses: List[str] = None,
        model_url: str = None
    ) -> Dict[str, Any]:
        """
        Run evaluation using a pre-existing API key (manual approach).
        
        This method assumes you've already created an API key in the Vijil dashboard
        and want to use it directly without auto-registration.
        
        Args:
            agent_name: Name for the agent in evaluation
            evaluation_name: Name for the evaluation run
            api_key_name: Name of pre-existing API key in Vijil
            harnesses: List of harnesses to run
            model_url: Custom model URL (if using custom endpoint)
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.vijil:
            self._init_vijil()
        
        if harnesses is None:
            harnesses = ["security"]  # Default harness
        
        print(f"\n🚀 Starting manual Vijil evaluation...")
        print(f"   Agent: {agent_name}")
        print(f"   Evaluation: {evaluation_name}")
        print(f"   API Key: {api_key_name}")
        print(f"   Harnesses: {', '.join(harnesses)}")
        
        try:
            # Create evaluation directly using existing API key
            print("🔄 Creating evaluation with existing API key...")
            evaluation = self.vijil.evaluations.create(
                model_hub="custom",
                model_name=agent_name,
                name=evaluation_name,
                api_key_name=api_key_name,
                model_url=model_url or "http://localhost:8000/v1",  # Default local URL
                harnesses=harnesses,
            )
            
            evaluation_id = evaluation.get("id")
            print(f"✅ Evaluation {evaluation_id} created successfully")
            print("📝 Note: This creates the evaluation but doesn't run a local server.")
            print("   You'll need to run your agent server separately if using a local model.")
            
            return {
                "evaluation_id": evaluation_id,
                "status": "created",
                "api_key_name": api_key_name,
                "results": evaluation
            }
            
        except Exception as e:
            error_msg = str(e)
            if "api_key_name" in error_msg.lower() or "not found" in error_msg.lower():
                print(f"❌ API key not found: {api_key_name}")
                print("   Please create this API key in the Vijil dashboard first:")
                print("   1. Go to Vijil dashboard")
                print("   2. Navigate to API Keys section")
                print(f"   3. Create a new API key named: {api_key_name}")
                print("   4. Make sure it has evaluation permissions")
            else:
                print(f"❌ Manual evaluation failed: {e}")
            raise
    
    def generate_evaluation_report(
        self,
        evaluation_id: str,
        report_format: str = 'html',
        report_file: Optional[str] = None,
        wait_till_completion: bool = True
    ) -> str:
        """
        Generate and download an evaluation report.
        
        Args:
            evaluation_id: The ID of the evaluation to generate a report for
            report_format: Format of the report ('html' or 'pdf')
            report_file: Optional filename to save the report (default: {evaluation_id}-report.{format})
            wait_till_completion: Whether to wait for completion (default: True)
            
        Returns:
            The filename of the generated report
        """
        if not self.vijil:
            self._init_vijil()
        
        # Set default filename if not provided
        if not report_file:
            report_file = f"{evaluation_id}-report.{report_format}"
        
        print(f"\n📊 Generating evaluation report...")
        print(f"   Evaluation ID: {evaluation_id}")
        print(f"   Format: {report_format.upper()}")
        print(f"   Output file: {report_file}")
        
        # Basic validation for evaluation ID
        if not evaluation_id or evaluation_id.strip() == "":
            print(f"❌ Invalid evaluation ID: '{evaluation_id}'")
            raise ValueError("Evaluation ID cannot be empty")
        
        try:
            # Get the report object
            report = self.vijil.evaluations.report(evaluation_id)
            
            # Generate the report
            report.generate(
                save_file=report_file,
                wait_till_completion=wait_till_completion
            )
            
            print(f"✅ Report generated successfully: {report_file}")
            return report_file
            
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "invalid" in error_msg.lower():
                print(f"❌ Evaluation not found: {evaluation_id}")
                print("   Make sure the evaluation ID is correct and the evaluation is completed")
            elif "unauthorized" in error_msg.lower():
                print(f"❌ Unauthorized access to evaluation: {evaluation_id}")
                print("   Make sure you have permission to access this evaluation")
            else:
                print(f"❌ Report generation failed: {e}")
            raise


def main():
    """CLI interface for running Vijil evaluations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Vijil evaluation on the AI Git Work Explainer agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run basic security evaluation
  %(prog)s --llm-provider anthropic           # Use Anthropic Claude for evaluation
  %(prog)s --harnesses security_Small ethics_Small  # Run multiple evaluations
  %(prog)s --advanced                         # Run advanced evaluation with monitoring
  %(prog)s --repo-path /path/to/repo          # Evaluate specific repository
  %(prog)s --generate-report abc123           # Generate HTML report for evaluation abc123
  %(prog)s --generate-report abc123 --report-format pdf  # Generate PDF report
  %(prog)s --simple-name                      # Use simple agent name (helps avoid API key issues)
  %(prog)s --api-key-name my-simple-key       # Use pre-existing API key from dashboard
  %(prog)s --test-api-key                     # Test if your VIJIL_API_KEY is working
  %(prog)s --check-setup                      # Check if environment is configured
  %(prog)s --list-harnesses                   # List available harnesses

Available Harnesses:
  security_Small    - Security-focused evaluation (default)
  ethics_Small      - Ethics and bias evaluation  
  privacy_Small     - Privacy protection evaluation
  toxicity_Small    - Toxicity evaluation
        """
    )
    
    parser.add_argument(
        '--repo-path',
        type=str,
        default='.',
        help='Path to git repository (default: current directory)'
    )
    
    parser.add_argument(
        '--harnesses',
        nargs='+',
        default=['security'],
        help='Evaluation harnesses to run (default: security)'
    )
    
    parser.add_argument(
        '--agent-name',
        type=str,
        default='Git Work Explainer',
        help='Name for the agent in evaluation'
    )
    
    parser.add_argument(
        '--evaluation-name',
        type=str,
        default='Local Agent Evaluation',
        help='Name for the evaluation run'
    )
    
    parser.add_argument(
        '--advanced',
        action='store_true',
        help='Run advanced evaluation with detailed monitoring'
    )
    
    parser.add_argument(
        '--rate-limit',
        type=int,
        default=30,
        help='Maximum requests per rate limit interval (default: 30)'
    )
    
    parser.add_argument(
        '--rate-limit-interval',
        type=int, 
        default=1,
        help='Rate limit interval in minutes (default: 1)'
    )
    
    parser.add_argument(
        '--check-setup',
        action='store_true',
        help='Check if environment is properly configured'
    )
    
    parser.add_argument(
        '--list-harnesses',
        action='store_true',
        help='List available evaluation harnesses'
    )
    
    parser.add_argument(
        '--test-agent',
        action='store_true',
        help='Test the agent function locally before running Vijil evaluation'
    )
    
    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['openai', 'anthropic'],
        default=None,
        help='LLM provider to use for evaluation (default: auto-detect available provider)'
    )
    
    parser.add_argument(
        '--generate-report',
        type=str,
        metavar='EVALUATION_ID',
        help='Generate and download evaluation report for the given evaluation ID'
    )
    
    parser.add_argument(
        '--report-format',
        type=str,
        choices=['html', 'pdf'],
        default='html',
        help='Report format (default: html)'
    )
    
    parser.add_argument(
        '--report-file',
        type=str,
        help='Output filename for the report (default: {evaluation_id}-report.{format})'
    )
    
    parser.add_argument(
        '--simple-name',
        action='store_true',
        help='Use a simple, predictable name for the agent (helpful for debugging API key issues)'
    )
    
    parser.add_argument(
        '--api-key-name',
        type=str,
        help='Use a specific API key name (must be pre-created in Vijil dashboard)'
    )
    
    parser.add_argument(
        '--test-api-key',
        action='store_true',
        help='Test if VIJIL_API_KEY is valid and working'
    )
    
    args = parser.parse_args()
    
    try:
        if args.list_harnesses:
            print("📋 Available Evaluation Harnesses:\n")
            harnesses = get_available_harnesses()
            harness_descriptions = {
                "security": "Security-focused evaluation",
                "ethics": "Ethics and bias evaluation",
                "trust_score": "General trustworthiness evaluation",
                "privacy": "Privacy protection evaluation (if available)", 
                "toxicity": "Toxicity and harmful content evaluation (if available)",
            }
            
            for harness in harnesses:
                description = harness_descriptions.get(harness, "Evaluation harness")
                print(f"   • {harness}")
                print(f"     {description}")
            
            print(f"\n💡 Usage: python vijil_executor.py --harnesses {' '.join(harnesses[:2])}")
            return 0
        
        if args.test_agent:
            print("🧪 Testing agent function locally...\n")
            
            # Initialize executor but don't require Vijil API key
            import asyncio
            executor = GitWorkExplainerExecutor(args.repo_path, args.llm_provider)
            
            # Test input adapter
            print("1. Testing input adapter...")
            try:
                from vijil.local_agents.models import ChatCompletionRequest
                mock_request = type('MockRequest', (), {
                    'messages': [{"role": "user", "content": "Analyze 3 commits for product managers"}]
                })()
                adapted_input = executor.input_adapter(mock_request)
                print("✅ Input adapter working")
            except Exception as e:
                print(f"❌ Input adapter failed: {e}")
                return 1
            
            # Test agent function
            print("2. Testing agent function...")
            try:
                result = asyncio.run(executor.agent_function(adapted_input))
                print("✅ Agent function working")
            except Exception as e:
                print(f"❌ Agent function failed: {e}")
                return 1
            
            # Test output adapter
            print("3. Testing output adapter...")
            try:
                response = executor.output_adapter(result)
                print("✅ Output adapter working")
                print(f"\n📝 Sample response content: {response.choices[0].message.content[:200]}...")
            except Exception as e:
                print(f"❌ Output adapter failed: {e}")
                return 1
            
            print("\n✅ All components working - agent is ready for Vijil evaluation!")
            return 0
        
        if args.check_setup:
            print("🔍 Checking Vijil evaluation setup...\n")
            
            # Check Vijil availability
            if not VIJIL_AVAILABLE:
                print("❌ Vijil SDK not installed")
                print("   Install with: pip install vijil")
                return 1
            else:
                print("✅ Vijil SDK is available")
            
            # Check environment variables
            vijil_ok = check_vijil_api_key()
            ngrok_ok = check_ngrok_auth()
            
            if not vijil_ok or not ngrok_ok:
                print("\n❌ Setup incomplete - please configure missing environment variables")
                return 1
            
            # Check agent initialization
            try:
                executor = GitWorkExplainerExecutor(args.repo_path, args.llm_provider)
                print("✅ Agent can be initialized successfully")
            except Exception as e:
                print(f"❌ Agent initialization failed: {e}")
                return 1
            
            print("\n✅ All checks passed - ready for evaluation!")
            return 0
        
        if args.test_api_key:
            print("🧪 Testing VIJIL_API_KEY...")
            
            # Check if API key is set
            api_key = os.getenv("VIJIL_API_KEY")
            if not api_key:
                print("❌ VIJIL_API_KEY is not set")
                print("   Please set your Vijil API key:")
                print("   export VIJIL_API_KEY=your_api_key_here")
                return 1
            
            print(f"✅ VIJIL_API_KEY is set (starts with: {api_key[:8]}...)")
            
            # Try to initialize Vijil client
            try:
                from vijil import Vijil
                vijil_client = Vijil(api_key=api_key)
                print("✅ Vijil client created successfully")
                
                # Try a simple API call - let's try getting evaluations
                print("🔍 Testing API authentication...")
                
                # This is a basic call that should work if authentication is good
                evaluations = vijil_client.evaluations.list(limit=1)  
                print("✅ API authentication successful!")
                print(f"   API response received (found {len(evaluations) if evaluations else 0} evaluations)")
                
                return 0
                
            except Exception as e:
                error_msg = str(e)
                if "unauthorized" in error_msg.lower() or "token validation" in error_msg.lower():
                    print("❌ API Key authentication failed")
                    print("   Your VIJIL_API_KEY appears to be invalid or expired")
                    print("   Please check your API key in the Vijil dashboard")
                elif "forbidden" in error_msg.lower():
                    print("❌ API Key permissions insufficient") 
                    print("   Your API key may not have the required permissions")
                else:
                    print(f"❌ API test failed: {e}")
                return 1
        
        if args.generate_report:
            print("📊 Generating evaluation report...\n")
            
            # Check for Vijil API key (needed for report generation)
            if not os.getenv("VIJIL_API_KEY"):
                print("❌ VIJIL_API_KEY is required for report generation")
                print("   Please set your Vijil API key in your environment variables:")
                print("   export VIJIL_API_KEY=your_api_key_here")
                return 1
            
            # Initialize evaluator for report generation (skip agent initialization)
            try:
                evaluator = VijilEvaluator(args.repo_path, args.llm_provider, skip_agent_init=True)
                report_file = evaluator.generate_evaluation_report(
                    evaluation_id=args.generate_report,
                    report_format=args.report_format,
                    report_file=args.report_file
                )
                print(f"\n📁 Report saved as: {report_file}")
                return 0
            except Exception as e:
                print(f"❌ Report generation failed: {e}")
                return 1
        
        # Initialize evaluator
        evaluator = VijilEvaluator(args.repo_path, args.llm_provider)
        
        # Use simple name if requested
        agent_name = args.agent_name
        if args.simple_name:
            agent_name = "git-work-explainer"  # Simple, predictable name
            print(f"📝 Using simple agent name: {agent_name}")
        
        # Run evaluation
        if args.api_key_name:
            # Use manual evaluation with pre-existing API key
            print("📝 Using manual evaluation with pre-existing API key")
            results = evaluator.run_manual_evaluation(
                agent_name=agent_name,
                evaluation_name=args.evaluation_name,
                api_key_name=args.api_key_name,
                harnesses=args.harnesses
            )
            print(f"\n📊 Evaluation Results:")
            print(json.dumps(results, indent=2))
        elif args.advanced:
            results = evaluator.run_advanced_evaluation(
                agent_name=agent_name,
                evaluation_name=args.evaluation_name,
                harnesses=args.harnesses,
                rate_limit=args.rate_limit,
                rate_limit_interval=args.rate_limit_interval
            )
            print(f"\n📊 Evaluation Results:")
            print(json.dumps(results, indent=2))
        else:
            # Use simple evaluation method (more reliable)
            print("📝 Using simple evaluation method (use --advanced for detailed monitoring)")
            evaluator.run_evaluation(
                agent_name=agent_name,
                evaluation_name=args.evaluation_name,
                harnesses=args.harnesses,
                rate_limit=args.rate_limit,
                rate_limit_interval=args.rate_limit_interval
            )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Evaluation cancelled by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
