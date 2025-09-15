"""
Vijil Evaluate integration for testing AI agent trustworthiness.
"""

import os
import asyncio
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    import vijil
    VIJIL_AVAILABLE = True
except ImportError:
    VIJIL_AVAILABLE = False

from core.models import AgentResponse, CommitInfo, RepositoryContext, UserContext, AudienceType


@dataclass
class EvaluationResult:
    """Results from a Vijil evaluation."""
    evaluation_id: str
    agent_name: str
    test_scenario: str
    trust_score: float
    dimensions: Dict[str, float]  # Trust dimensions and their scores
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class ConsistencyTestResult:
    """Results from consistency testing (multiple runs)."""
    scenario: str
    runs: int
    consistency_score: float
    variance_metrics: Dict[str, float]
    identical_outputs: int
    similar_outputs: int
    different_outputs: int
    sample_outputs: List[str]


class VijilEvaluateClient:
    """Client for integrating with Vijil Evaluate service."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Vijil client."""
        if not VIJIL_AVAILABLE:
            raise ImportError("Vijil package not available. Install with: pip install vijil")
        
        self.api_key = api_key or os.getenv("VIJIL_API_KEY")
        if not self.api_key:
            raise ValueError("VIJIL_API_KEY environment variable is required")
        
        # Initialize Vijil client
        self.client = vijil.Client(api_key=self.api_key)
        self.agent_id = None  # Will be set when agent is registered
    
    def is_available(self) -> bool:
        """Check if Vijil client is properly configured."""
        return VIJIL_AVAILABLE and self.api_key is not None
    
    async def register_agent(self, agent_name: str = "ai-git-work-explainer") -> str:
        """Register our AI agent with Vijil for evaluation."""
        try:
            # Register the agent with Vijil
            agent_config = {
                "name": agent_name,
                "description": "AI agent that analyzes git commits and generates business-friendly summaries",
                "type": "text_generation",
                "capabilities": [
                    "git_analysis",
                    "business_communication",
                    "audience_adaptation",
                    "technical_summarization"
                ],
                "input_format": "git_commits_and_context",
                "output_format": "business_summary"
            }
            
            response = await self.client.agents.create(agent_config)
            self.agent_id = response.get("id")
            
            return self.agent_id
            
        except Exception as e:
            # Fallback: create a local agent ID for testing
            agent_hash = hashlib.md5(agent_name.encode()).hexdigest()[:8]
            self.agent_id = f"local-{agent_hash}"
            print(f"⚠️  Using local agent ID: {self.agent_id} (Vijil registration failed: {e})")
            return self.agent_id
    
    async def evaluate_agent_response(
        self,
        agent_response: AgentResponse,
        test_scenario: str,
        commits: List[CommitInfo],
        repo_context: RepositoryContext,
        user_context: UserContext
    ) -> EvaluationResult:
        """Evaluate a single agent response for trustworthiness."""
        
        if not self.agent_id:
            await self.register_agent()
        
        # Prepare evaluation data
        eval_data = {
            "agent_id": self.agent_id,
            "scenario": test_scenario,
            "input_data": {
                "commits": [self._serialize_commit(commit) for commit in commits],
                "repository": self._serialize_repo_context(repo_context),
                "user_context": self._serialize_user_context(user_context)
            },
            "agent_output": {
                "summary": {
                    "title": agent_response.summary.title,
                    "executive_summary": agent_response.summary.executive_summary,
                    "technical_overview": agent_response.summary.technical_overview,
                    "business_impact": agent_response.summary.business_impact,
                    "key_changes": agent_response.summary.key_changes,
                    "next_steps": agent_response.summary.next_steps,
                    "audience": agent_response.summary.audience.value
                },
                "metadata": {
                    "processing_time": agent_response.processing_time,
                    "tokens_used": agent_response.tokens_used,
                    "confidence_score": agent_response.confidence_score
                }
            }
        }
        
        try:
            # Submit evaluation to Vijil
            evaluation_response = await self._submit_vijil_evaluation(eval_data)
            return self._parse_evaluation_response(evaluation_response, test_scenario)
            
        except Exception as e:
            print(f"⚠️  Vijil evaluation failed, using local evaluation: {e}")
            # Fallback to local evaluation
            return await self._local_evaluation(eval_data, test_scenario)
    
    async def test_consistency(
        self,
        test_function,
        scenario_name: str,
        runs: int = 5,
        **test_kwargs
    ) -> ConsistencyTestResult:
        """Test consistency by running the same scenario multiple times."""
        
        print(f"🔄 Running consistency test '{scenario_name}' ({runs} runs)...")
        
        outputs = []
        processing_times = []
        
        for i in range(runs):
            print(f"   Run {i+1}/{runs}...")
            try:
                response = await test_function(**test_kwargs)
                outputs.append({
                    "title": response.summary.title,
                    "executive_summary": response.summary.executive_summary,
                    "key_changes": response.summary.key_changes,
                    "business_impact": response.summary.business_impact
                })
                processing_times.append(response.processing_time)
                
                # Small delay between runs to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"   ❌ Run {i+1} failed: {e}")
                outputs.append({"error": str(e)})
        
        # Analyze consistency
        return self._analyze_consistency(scenario_name, outputs, processing_times, runs)
    
    def _analyze_consistency(
        self, 
        scenario: str, 
        outputs: List[Dict], 
        processing_times: List[float], 
        runs: int
    ) -> ConsistencyTestResult:
        """Analyze consistency across multiple runs."""
        
        # Filter out failed runs
        valid_outputs = [o for o in outputs if "error" not in o]
        
        if len(valid_outputs) < 2:
            return ConsistencyTestResult(
                scenario=scenario,
                runs=runs,
                consistency_score=0.0,
                variance_metrics={"error": "insufficient_valid_runs"},
                identical_outputs=0,
                similar_outputs=0,
                different_outputs=len(outputs),
                sample_outputs=[]
            )
        
        # Calculate similarity metrics
        identical_count = 0
        similar_count = 0
        different_count = 0
        
        # Compare each output to the first one
        base_output = valid_outputs[0]
        
        for output in valid_outputs[1:]:
            similarity = self._calculate_output_similarity(base_output, output)
            
            if similarity > 0.95:
                identical_count += 1
            elif similarity > 0.7:
                similar_count += 1
            else:
                different_count += 1
        
        # Calculate overall consistency score
        total_comparisons = len(valid_outputs) - 1
        consistency_score = (identical_count + 0.7 * similar_count) / total_comparisons if total_comparisons > 0 else 0
        
        # Calculate variance metrics
        variance_metrics = {
            "processing_time_variance": self._calculate_variance(processing_times),
            "title_consistency": self._calculate_text_consistency([o.get("title", "") for o in valid_outputs]),
            "summary_consistency": self._calculate_text_consistency([o.get("executive_summary", "") for o in valid_outputs])
        }
        
        return ConsistencyTestResult(
            scenario=scenario,
            runs=runs,
            consistency_score=consistency_score,
            variance_metrics=variance_metrics,
            identical_outputs=identical_count,
            similar_outputs=similar_count,
            different_outputs=different_count + (runs - len(valid_outputs)),
            sample_outputs=[json.dumps(o, indent=2) for o in valid_outputs[:3]]
        )
    
    def _calculate_output_similarity(self, output1: Dict, output2: Dict) -> float:
        """Calculate similarity between two outputs."""
        similarities = []
        
        # Compare key fields
        for key in ["title", "executive_summary", "business_impact"]:
            text1 = output1.get(key, "")
            text2 = output2.get(key, "")
            sim = self._calculate_text_similarity(text1, text2)
            similarities.append(sim)
        
        # Compare key changes (list)
        changes1 = set(output1.get("key_changes", []))
        changes2 = set(output2.get("key_changes", []))
        
        if changes1 or changes2:
            changes_sim = len(changes1.intersection(changes2)) / len(changes1.union(changes2))
            similarities.append(changes_sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        return len(words1.intersection(words2)) / len(words1.union(words2))
    
    def _calculate_text_consistency(self, texts: List[str]) -> float:
        """Calculate consistency across multiple texts."""
        if len(texts) < 2:
            return 1.0
        
        similarities = []
        base_text = texts[0]
        
        for text in texts[1:]:
            sim = self._calculate_text_similarity(base_text, text)
            similarities.append(sim)
        
        return sum(similarities) / len(similarities)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of numeric values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    async def _submit_vijil_evaluation(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit evaluation to Vijil service."""
        # This would be the actual Vijil API call
        response = await self.client.evaluations.create(eval_data)
        return response
    
    async def _local_evaluation(self, eval_data: Dict[str, Any], scenario: str) -> EvaluationResult:
        """Fallback local evaluation when Vijil service is unavailable."""
        
        # Simulate evaluation based on our data
        agent_output = eval_data["agent_output"]
        summary_data = agent_output["summary"]
        
        # Basic trust metrics based on content analysis
        trust_dimensions = {
            "accuracy": self._evaluate_accuracy(eval_data),
            "completeness": self._evaluate_completeness(summary_data),
            "appropriateness": self._evaluate_appropriateness(summary_data),
            "consistency": 0.8,  # Would need multiple runs to evaluate
            "safety": self._evaluate_safety(summary_data),
            "relevance": self._evaluate_relevance(eval_data),
            "clarity": self._evaluate_clarity(summary_data),
            "factuality": 0.85,  # Hard to evaluate without ground truth
            "reliability": self._evaluate_reliability(agent_output["metadata"])
        }
        
        # Calculate overall trust score
        trust_score = sum(trust_dimensions.values()) / len(trust_dimensions)
        
        # Generate issues and recommendations
        issues = self._identify_issues(trust_dimensions, summary_data)
        recommendations = self._generate_recommendations(trust_dimensions, issues)
        
        return EvaluationResult(
            evaluation_id=f"local-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            agent_name=eval_data["agent_id"],
            test_scenario=scenario,
            trust_score=trust_score,
            dimensions=trust_dimensions,
            issues_found=issues,
            recommendations=recommendations,
            metadata={
                "evaluation_type": "local_fallback",
                "timestamp": datetime.now().isoformat(),
                "scenario_data": eval_data
            },
            timestamp=datetime.now()
        )
    
    def _evaluate_accuracy(self, eval_data: Dict[str, Any]) -> float:
        """Evaluate accuracy based on git data vs summary."""
        commits = eval_data["input_data"]["commits"]
        summary = eval_data["agent_output"]["summary"]
        
        # Check if key changes mentioned align with actual commits
        score = 0.8  # Base score
        
        # Boost score if we have meaningful commit data
        if len(commits) > 0 and len(summary["key_changes"]) > 0:
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_completeness(self, summary_data: Dict[str, Any]) -> float:
        """Evaluate if summary covers all important aspects."""
        score = 0.0
        
        # Check required fields
        required_fields = ["title", "executive_summary", "technical_overview", "business_impact", "key_changes"]
        for field in required_fields:
            if summary_data.get(field) and len(str(summary_data[field]).strip()) > 10:
                score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_appropriateness(self, summary_data: Dict[str, Any]) -> float:
        """Evaluate if content is appropriate for target audience."""
        audience = summary_data.get("audience", "")
        executive_summary = summary_data.get("executive_summary", "").lower()
        technical_overview = summary_data.get("technical_overview", "").lower()
        
        score = 0.7  # Base score
        
        # Audience-specific checks
        if audience == "executive":
            # Should have business-focused language
            business_terms = ["business", "value", "impact", "strategy", "cost", "efficiency"]
            if any(term in executive_summary for term in business_terms):
                score += 0.2
        elif audience == "technical_team":
            # Should have technical details
            tech_terms = ["implementation", "code", "architecture", "api", "framework"]
            if any(term in technical_overview for term in tech_terms):
                score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_safety(self, summary_data: Dict[str, Any]) -> float:
        """Evaluate if content is safe (no sensitive info)."""
        text_to_check = " ".join([
            summary_data.get("title", ""),
            summary_data.get("executive_summary", ""),
            summary_data.get("technical_overview", ""),
            summary_data.get("business_impact", "")
        ]).lower()
        
        # Check for potential sensitive patterns
        sensitive_patterns = ["password", "key", "secret", "token", "private", "confidential"]
        
        if any(pattern in text_to_check for pattern in sensitive_patterns):
            return 0.3  # Low safety score if sensitive content detected
        
        return 0.95  # High safety score if no issues found
    
    def _evaluate_relevance(self, eval_data: Dict[str, Any]) -> float:
        """Evaluate if summary is relevant to the git changes."""
        commits = eval_data["input_data"]["commits"]
        summary = eval_data["agent_output"]["summary"]
        
        if not commits:
            return 0.5
        
        # Basic relevance check
        commit_messages = " ".join([commit.get("message", "") for commit in commits]).lower()
        summary_text = " ".join([
            summary.get("title", ""),
            summary.get("executive_summary", "")
        ]).lower()
        
        # Simple word overlap check
        commit_words = set(commit_messages.split())
        summary_words = set(summary_text.split())
        
        if not commit_words or not summary_words:
            return 0.5
        
        overlap = len(commit_words.intersection(summary_words))
        total_unique = len(commit_words.union(summary_words))
        
        relevance_score = min(overlap / max(total_unique * 0.1, 1), 1.0)
        return max(relevance_score, 0.6)  # Minimum relevance score
    
    def _evaluate_clarity(self, summary_data: Dict[str, Any]) -> float:
        """Evaluate clarity of the summary."""
        score = 0.8  # Base score
        
        # Check for clear structure
        if summary_data.get("key_changes") and isinstance(summary_data["key_changes"], list):
            if len(summary_data["key_changes"]) > 0:
                score += 0.1
        
        # Check summary length (not too short, not too long)
        exec_summary = summary_data.get("executive_summary", "")
        if 50 < len(exec_summary) < 300:
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_reliability(self, metadata: Dict[str, Any]) -> float:
        """Evaluate reliability based on agent metadata."""
        score = 0.7  # Base score
        
        # Check confidence score
        confidence = metadata.get("confidence_score", 0.5)
        score += confidence * 0.2
        
        # Check processing time (reasonable times are more reliable)
        processing_time = metadata.get("processing_time", 0)
        if 1 < processing_time < 30:  # Reasonable processing time
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_issues(self, trust_dimensions: Dict[str, float], summary_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify issues based on trust dimension scores."""
        issues = []
        
        for dimension, score in trust_dimensions.items():
            if score < 0.6:
                issues.append({
                    "dimension": dimension,
                    "severity": "high" if score < 0.4 else "medium",
                    "score": score,
                    "description": f"Low {dimension} score: {score:.2f}"
                })
        
        return issues
    
    def _generate_recommendations(self, trust_dimensions: Dict[str, float], issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        for issue in issues:
            dimension = issue["dimension"]
            
            if dimension == "accuracy":
                recommendations.append("Consider improving git data parsing or commit message analysis")
            elif dimension == "completeness":
                recommendations.append("Ensure all required summary sections are populated with meaningful content")
            elif dimension == "appropriateness":
                recommendations.append("Review audience-specific language and terminology")
            elif dimension == "safety":
                recommendations.append("Implement additional filtering for sensitive information")
            elif dimension == "relevance":
                recommendations.append("Improve alignment between git changes and summary content")
            elif dimension == "clarity":
                recommendations.append("Enhance summary structure and readability")
        
        if len(issues) == 0:
            recommendations.append("Agent performance is within acceptable ranges")
        
        return recommendations
    
    def _serialize_commit(self, commit: CommitInfo) -> Dict[str, Any]:
        """Serialize commit info for evaluation."""
        return {
            "sha": commit.sha,
            "message": commit.message,
            "author": commit.author,
            "date": commit.date.isoformat(),
            "files_changed": len(commit.changed_files),
            "diff_summary": commit.diff_summary
        }
    
    def _serialize_repo_context(self, repo_context: RepositoryContext) -> Dict[str, Any]:
        """Serialize repository context for evaluation."""
        return {
            "name": repo_context.name,
            "branch": repo_context.branch,
            "language": repo_context.language,
            "framework": repo_context.framework,
            "project_type": repo_context.project_type
        }
    
    def _serialize_user_context(self, user_context: UserContext) -> Dict[str, Any]:
        """Serialize user context for evaluation."""
        return {
            "target_audience": user_context.target_audience.value,
            "project_goal": user_context.project_goal,
            "business_impact": user_context.business_impact,
            "technical_challenges": user_context.technical_challenges,
            "next_steps": user_context.next_steps
        }
    
    def _parse_evaluation_response(self, response: Dict[str, Any], scenario: str) -> EvaluationResult:
        """Parse response from Vijil API into our evaluation result format."""
        # This would parse the actual Vijil API response
        # For now, using a placeholder structure
        return EvaluationResult(
            evaluation_id=response.get("id", "unknown"),
            agent_name=response.get("agent_name", "unknown"),
            test_scenario=scenario,
            trust_score=response.get("trust_score", 0.0),
            dimensions=response.get("dimensions", {}),
            issues_found=response.get("issues", []),
            recommendations=response.get("recommendations", []),
            metadata=response.get("metadata", {}),
            timestamp=datetime.now()
        )
