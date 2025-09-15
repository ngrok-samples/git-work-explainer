"""
Evaluation harnesses for testing specific scenarios with Vijil Evaluate.
These test our AI agent's behavior across different git repository patterns and use cases.
"""

import asyncio
import tempfile
import shutil
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import git

from core.models import AudienceType, UserContext
from core.agent import WorkExplainerAgent
from core.vijil_client import VijilEvaluateClient, EvaluationResult, ConsistencyTestResult


class EvaluationHarness:
    """Main harness for running comprehensive evaluations of the AI agent."""
    
    def __init__(self, agent: WorkExplainerAgent, vijil_client: VijilEvaluateClient):
        self.agent = agent
        self.vijil_client = vijil_client
        self.test_repos: Dict[str, Path] = {}
    
    async def run_full_evaluation_suite(self) -> Dict[str, Any]:
        """Run the complete evaluation suite."""
        print("🧪 Starting comprehensive evaluation suite...")
        
        results = {
            "timestamp": "2024-01-01T00:00:00Z",  # Will be set properly
            "scenarios": {},
            "consistency_tests": {},
            "overall_metrics": {},
            "recommendations": []
        }
        
        try:
            # Create test repositories
            await self._setup_test_repositories()
            
            # Run scenario-based evaluations
            scenario_results = await self._run_scenario_evaluations()
            results["scenarios"] = scenario_results
            
            # Run consistency tests
            consistency_results = await self._run_consistency_evaluations()
            results["consistency_tests"] = consistency_results
            
            # Calculate overall metrics
            results["overall_metrics"] = self._calculate_overall_metrics(scenario_results, consistency_results)
            
            # Generate recommendations
            results["recommendations"] = self._generate_overall_recommendations(results)
            
            print("✅ Evaluation suite completed successfully")
            
        except Exception as e:
            print(f"❌ Evaluation suite failed: {e}")
            results["error"] = str(e)
        
        finally:
            # Clean up test repositories
            await self._cleanup_test_repositories()
        
        return results
    
    async def evaluate_specific_scenario(self, scenario_name: str, **kwargs) -> EvaluationResult:
        """Run a specific evaluation scenario."""
        scenarios = {
            "basic_feature_development": self._test_basic_feature_development,
            "bug_fix_analysis": self._test_bug_fix_analysis,
            "large_refactor": self._test_large_refactor,
            "mixed_work_types": self._test_mixed_work_types,
            "audience_adaptation": self._test_audience_adaptation,
            "minimal_commits": self._test_minimal_commits,
            "complex_repository": self._test_complex_repository
        }
        
        if scenario_name not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        print(f"🎯 Running scenario: {scenario_name}")
        
        try:
            await self._setup_test_repositories()
            result = await scenarios[scenario_name](**kwargs)
            return result
        finally:
            await self._cleanup_test_repositories()
    
    async def test_consistency(self, scenario_name: str, runs: int = 5) -> ConsistencyTestResult:
        """Test consistency of a specific scenario."""
        if scenario_name == "basic_feature_development":
            test_function = self._run_basic_feature_test
        elif scenario_name == "executive_summary":
            test_function = self._run_executive_summary_test
        else:
            raise ValueError(f"Consistency test not available for scenario: {scenario_name}")
        
        return await self.vijil_client.test_consistency(
            test_function=test_function,
            scenario_name=scenario_name,
            runs=runs
        )
    
    async def _setup_test_repositories(self):
        """Create test git repositories for evaluation."""
        print("🏗️  Setting up test repositories...")
        
        # Test repo 1: Web application with feature development
        self.test_repos["web_app"] = await self._create_web_app_repo()
        
        # Test repo 2: API service with bug fixes
        self.test_repos["api_service"] = await self._create_api_service_repo()
        
        # Test repo 3: Library with refactoring
        self.test_repos["library"] = await self._create_library_repo()
    
    async def _create_web_app_repo(self) -> Path:
        """Create a test web application repository."""
        repo_path = Path(tempfile.mkdtemp(prefix="test_web_app_"))
        
        # Initialize git repo
        repo = git.Repo.init(repo_path)
        
        # Create initial structure
        (repo_path / "src").mkdir()
        (repo_path / "src" / "components").mkdir()
        (repo_path / "public").mkdir()
        
        # Initial commit
        (repo_path / "package.json").write_text('''{
  "name": "test-web-app",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.0.0"
  }
}''')
        
        (repo_path / "src" / "App.js").write_text('''import React from 'react';

function App() {
  return <div>Hello World</div>;
}

export default App;
''')
        
        repo.index.add_to_index([str(repo_path / "package.json"), str(repo_path / "src" / "App.js")])
        repo.index.commit("Initial commit: Set up React application")
        
        # Feature development commits
        (repo_path / "src" / "components" / "UserAuth.js").write_text('''import React, { useState } from 'react';

export default function UserAuth() {
  const [user, setUser] = useState(null);
  
  const login = async (credentials) => {
    // Login implementation
    setUser(credentials.username);
  };
  
  return (
    <div>
      {user ? `Welcome ${user}` : 'Please login'}
    </div>
  );
}
''')
        
        repo.index.add_to_index([str(repo_path / "src" / "components" / "UserAuth.js")])
        repo.index.commit("Add user authentication component")
        
        # API integration
        (repo_path / "src" / "api" / "auth.js").mkdir(parents=True)
        (repo_path / "src" / "api" / "auth.js").write_text('''const API_BASE = process.env.REACT_APP_API_URL;

export async function authenticate(username, password) {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });
  
  return response.json();
}
''')
        
        repo.index.add_to_index([str(repo_path / "src" / "api" / "auth.js")])
        repo.index.commit("Implement authentication API integration")
        
        # UI improvements
        (repo_path / "src" / "styles" / "auth.css").mkdir(parents=True)
        (repo_path / "src" / "styles" / "auth.css").write_text('''.auth-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
}

.auth-form {
  background: white;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
''')
        
        repo.index.add_to_index([str(repo_path / "src" / "styles" / "auth.css")])
        repo.index.commit("Add authentication UI styling")
        
        return repo_path
    
    async def _create_api_service_repo(self) -> Path:
        """Create a test API service repository with bug fixes."""
        repo_path = Path(tempfile.mkdtemp(prefix="test_api_service_"))
        
        repo = git.Repo.init(repo_path)
        
        # Initial API structure
        (repo_path / "requirements.txt").write_text('''fastapi==0.68.0
uvicorn==0.15.0
sqlalchemy==1.4.0
''')
        
        (repo_path / "main.py").write_text('''from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # Bug: No validation
    return {"user_id": user_id, "name": f"User {user_id}"}
''')
        
        repo.index.add_to_index([str(repo_path / "requirements.txt"), str(repo_path / "main.py")])
        repo.index.commit("Initial API setup")
        
        # Bug fix commits
        (repo_path / "main.py").write_text('''from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    return {"user_id": user_id, "name": f"User {user_id}"}
''')
        
        repo.index.add_to_index([str(repo_path / "main.py")])
        repo.index.commit("Fix: Add user ID validation")
        
        # Security improvement
        (repo_path / "auth.py").write_text('''import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"

def create_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token: str) -> int:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except jwt.PyJWTError:
        return None
''')
        
        repo.index.add_to_index([str(repo_path / "auth.py")])
        repo.index.commit("Add JWT authentication system")
        
        return repo_path
    
    async def _create_library_repo(self) -> Path:
        """Create a test library repository with refactoring."""
        repo_path = Path(tempfile.mkdtemp(prefix="test_library_"))
        
        repo = git.Repo.init(repo_path)
        
        # Initial library structure
        (repo_path / "setup.py").write_text('''from setuptools import setup, find_packages

setup(
    name="test-library",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
    ]
)
''')
        
        (repo_path / "testlib" / "__init__.py").mkdir(parents=True)
        (repo_path / "testlib" / "__init__.py").write_text('')
        
        (repo_path / "testlib" / "client.py").write_text('''import requests

class ApiClient:
    def __init__(self, base_url):
        self.base_url = base_url
    
    def get(self, endpoint):
        response = requests.get(f"{self.base_url}/{endpoint}")
        return response.json()
    
    def post(self, endpoint, data):
        response = requests.post(f"{self.base_url}/{endpoint}", json=data)
        return response.json()
''')
        
        repo.index.add_to_index([
            str(repo_path / "setup.py"),
            str(repo_path / "testlib" / "__init__.py"),
            str(repo_path / "testlib" / "client.py")
        ])
        repo.index.commit("Initial library implementation")
        
        # Refactoring commits
        (repo_path / "testlib" / "exceptions.py").write_text('''class ApiError(Exception):
    def __init__(self, message, status_code=None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class NotFoundError(ApiError):
    pass

class ValidationError(ApiError):
    pass
''')
        
        repo.index.add_to_index([str(repo_path / "testlib" / "exceptions.py")])
        repo.index.commit("Refactor: Add custom exception classes")
        
        # Improved client with error handling
        (repo_path / "testlib" / "client.py").write_text('''import requests
from .exceptions import ApiError, NotFoundError, ValidationError

class ApiClient:
    def __init__(self, base_url, timeout=30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    def _handle_response(self, response):
        if response.status_code == 404:
            raise NotFoundError("Resource not found", response.status_code)
        elif response.status_code == 400:
            raise ValidationError("Invalid request", response.status_code)
        elif not response.ok:
            raise ApiError(f"API error: {response.status_code}", response.status_code)
        
        return response.json()
    
    def get(self, endpoint):
        response = requests.get(
            f"{self.base_url}/{endpoint.lstrip('/')}", 
            timeout=self.timeout
        )
        return self._handle_response(response)
    
    def post(self, endpoint, data):
        response = requests.post(
            f"{self.base_url}/{endpoint.lstrip('/')}", 
            json=data,
            timeout=self.timeout
        )
        return self._handle_response(response)
''')
        
        repo.index.add_to_index([str(repo_path / "testlib" / "client.py")])
        repo.index.commit("Refactor: Improve error handling and add timeout support")
        
        return repo_path
    
    async def _run_scenario_evaluations(self) -> Dict[str, EvaluationResult]:
        """Run all scenario-based evaluations."""
        scenarios = {
            "basic_feature_development": self._test_basic_feature_development,
            "bug_fix_analysis": self._test_bug_fix_analysis,
            "large_refactor": self._test_large_refactor,
            "audience_adaptation": self._test_audience_adaptation
        }
        
        results = {}
        
        for scenario_name, test_function in scenarios.items():
            print(f"🎯 Testing scenario: {scenario_name}")
            try:
                result = await test_function()
                results[scenario_name] = result
                print(f"   ✅ Trust score: {result.trust_score:.2f}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[scenario_name] = {"error": str(e)}
        
        return results
    
    async def _run_consistency_evaluations(self) -> Dict[str, ConsistencyTestResult]:
        """Run consistency tests."""
        consistency_tests = {
            "basic_feature_development": 3,
            "executive_summary": 3
        }
        
        results = {}
        
        for test_name, runs in consistency_tests.items():
            print(f"🔄 Testing consistency: {test_name}")
            try:
                result = await self.test_consistency(test_name, runs)
                results[test_name] = result
                print(f"   ✅ Consistency score: {result.consistency_score:.2f}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[test_name] = {"error": str(e)}
        
        return results
    
    # Individual test scenarios
    
    async def _test_basic_feature_development(self) -> EvaluationResult:
        """Test analysis of basic feature development work."""
        repo_path = self.test_repos["web_app"]
        
        user_context = UserContext(
            target_audience=AudienceType.PRODUCT_MANAGER,
            project_goal="Implement user authentication system",
            business_impact="Enable personalized user experiences and improve security"
        )
        
        response = await self.agent.explain_work(
            commit_count=3,
            audience=AudienceType.PRODUCT_MANAGER,
            repo_path=str(repo_path),
            interactive=False,
            user_context=user_context
        )
        
        # Get repository context for evaluation
        self.agent.set_repository(str(repo_path))
        commits = self.agent.git_analyzer.get_recent_commits(3)
        repo_context = self.agent.git_analyzer.get_repository_context()
        
        return await self.vijil_client.evaluate_agent_response(
            agent_response=response,
            test_scenario="basic_feature_development",
            commits=commits,
            repo_context=repo_context,
            user_context=user_context
        )
    
    async def _test_bug_fix_analysis(self) -> EvaluationResult:
        """Test analysis of bug fix work."""
        repo_path = self.test_repos["api_service"]
        
        user_context = UserContext(
            target_audience=AudienceType.ENGINEERING_LEADERSHIP,
            project_goal="Fix security vulnerabilities and improve API reliability",
            technical_challenges="Balancing security with performance"
        )
        
        response = await self.agent.explain_work(
            commit_count=3,
            audience=AudienceType.ENGINEERING_LEADERSHIP,
            repo_path=str(repo_path),
            interactive=False,
            user_context=user_context
        )
        
        self.agent.set_repository(str(repo_path))
        commits = self.agent.git_analyzer.get_recent_commits(3)
        repo_context = self.agent.git_analyzer.get_repository_context()
        
        return await self.vijil_client.evaluate_agent_response(
            agent_response=response,
            test_scenario="bug_fix_analysis",
            commits=commits,
            repo_context=repo_context,
            user_context=user_context
        )
    
    async def _test_large_refactor(self) -> EvaluationResult:
        """Test analysis of refactoring work."""
        repo_path = self.test_repos["library"]
        
        user_context = UserContext(
            target_audience=AudienceType.TECHNICAL_TEAM,
            project_goal="Improve code maintainability and error handling",
            technical_challenges="Maintaining backward compatibility while improving architecture"
        )
        
        response = await self.agent.explain_work(
            commit_count=3,
            audience=AudienceType.TECHNICAL_TEAM,
            repo_path=str(repo_path),
            interactive=False,
            user_context=user_context
        )
        
        self.agent.set_repository(str(repo_path))
        commits = self.agent.git_analyzer.get_recent_commits(3)
        repo_context = self.agent.git_analyzer.get_repository_context()
        
        return await self.vijil_client.evaluate_agent_response(
            agent_response=response,
            test_scenario="large_refactor",
            commits=commits,
            repo_context=repo_context,
            user_context=user_context
        )
    
    async def _test_audience_adaptation(self) -> EvaluationResult:
        """Test adaptation to different audiences."""
        repo_path = self.test_repos["web_app"]
        
        # Test with executive audience
        user_context = UserContext(
            target_audience=AudienceType.EXECUTIVE,
            project_goal="Increase user engagement and revenue through better authentication",
            business_impact="Expected 15% increase in user retention"
        )
        
        response = await self.agent.explain_work(
            commit_count=3,
            audience=AudienceType.EXECUTIVE,
            repo_path=str(repo_path),
            interactive=False,
            user_context=user_context
        )
        
        self.agent.set_repository(str(repo_path))
        commits = self.agent.git_analyzer.get_recent_commits(3)
        repo_context = self.agent.git_analyzer.get_repository_context()
        
        return await self.vijil_client.evaluate_agent_response(
            agent_response=response,
            test_scenario="audience_adaptation",
            commits=commits,
            repo_context=repo_context,
            user_context=user_context
        )
    
    # Consistency test functions
    
    async def _run_basic_feature_test(self):
        """Basic feature test for consistency checking."""
        return await self._test_basic_feature_development()
    
    async def _run_executive_summary_test(self):
        """Executive summary test for consistency checking."""
        return await self._test_audience_adaptation()
    
    # Utility methods
    
    def _calculate_overall_metrics(self, scenario_results: Dict, consistency_results: Dict) -> Dict[str, float]:
        """Calculate overall metrics across all tests."""
        metrics = {
            "average_trust_score": 0.0,
            "average_consistency_score": 0.0,
            "pass_rate": 0.0,
            "total_tests": 0,
            "passed_tests": 0
        }
        
        # Calculate average trust score
        trust_scores = []
        for scenario, result in scenario_results.items():
            if hasattr(result, 'trust_score'):
                trust_scores.append(result.trust_score)
                metrics["total_tests"] += 1
                if result.trust_score >= 0.7:
                    metrics["passed_tests"] += 1
        
        if trust_scores:
            metrics["average_trust_score"] = sum(trust_scores) / len(trust_scores)
        
        # Calculate average consistency score
        consistency_scores = []
        for test, result in consistency_results.items():
            if hasattr(result, 'consistency_score'):
                consistency_scores.append(result.consistency_score)
        
        if consistency_scores:
            metrics["average_consistency_score"] = sum(consistency_scores) / len(consistency_scores)
        
        # Calculate pass rate
        if metrics["total_tests"] > 0:
            metrics["pass_rate"] = metrics["passed_tests"] / metrics["total_tests"]
        
        return metrics
    
    def _generate_overall_recommendations(self, results: Dict) -> List[str]:
        """Generate overall recommendations based on all test results."""
        recommendations = []
        
        overall_metrics = results.get("overall_metrics", {})
        
        if overall_metrics.get("average_trust_score", 0) < 0.7:
            recommendations.append("Consider improving overall agent reliability - trust score below threshold")
        
        if overall_metrics.get("average_consistency_score", 0) < 0.8:
            recommendations.append("Work on improving response consistency across multiple runs")
        
        if overall_metrics.get("pass_rate", 0) < 0.8:
            recommendations.append("Focus on scenarios that are failing evaluations")
        
        # Add scenario-specific recommendations
        scenario_results = results.get("scenarios", {})
        for scenario, result in scenario_results.items():
            if hasattr(result, 'trust_score') and result.trust_score < 0.6:
                recommendations.append(f"Address issues in {scenario} scenario - low trust score")
        
        if not recommendations:
            recommendations.append("Agent performance is within acceptable ranges across all tests")
        
        return recommendations
    
    async def _cleanup_test_repositories(self):
        """Clean up temporary test repositories."""
        print("🧹 Cleaning up test repositories...")
        
        for repo_name, repo_path in self.test_repos.items():
            try:
                if repo_path.exists():
                    shutil.rmtree(repo_path)
            except Exception as e:
                print(f"   ⚠️  Failed to cleanup {repo_name}: {e}")
        
        self.test_repos.clear()


# Convenience functions for easy testing

async def run_quick_evaluation(agent: WorkExplainerAgent) -> Dict[str, Any]:
    """Run a quick evaluation with basic scenarios."""
    vijil_client = VijilEvaluateClient()
    harness = EvaluationHarness(agent, vijil_client)
    
    # Run just a couple of key scenarios
    results = {}
    
    try:
        await harness._setup_test_repositories()
        results["basic_feature"] = await harness._test_basic_feature_development()
        results["consistency"] = await harness.test_consistency("basic_feature_development", runs=2)
    finally:
        await harness._cleanup_test_repositories()
    
    return results


async def run_scenario_test(agent: WorkExplainerAgent, scenario_name: str) -> EvaluationResult:
    """Run a specific scenario test."""
    vijil_client = VijilEvaluateClient()
    harness = EvaluationHarness(agent, vijil_client)
    
    return await harness.evaluate_specific_scenario(scenario_name)
