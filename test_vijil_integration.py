#!/usr/bin/env python3
"""
Test script for Vijil integration

This script tests the Vijil integration components without requiring API keys.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from vijil_executor import GitWorkExplainerExecutor, VijilEvaluator, check_ngrok_auth, check_vijil_api_key
    VIJIL_EXECUTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import vijil_executor: {e}")
    VIJIL_EXECUTOR_AVAILABLE = False

from core.agent import WorkExplainerAgent
from core.models import AudienceType


class TestVijilIntegration(unittest.TestCase):
    """Test the Vijil integration components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_repo_path = "."
        
    @unittest.skipUnless(VIJIL_EXECUTOR_AVAILABLE, "vijil_executor not available")
    def test_git_work_explainer_executor_init(self):
        """Test GitWorkExplainerExecutor initialization."""
        with patch('vijil_executor.WorkExplainerAgent') as mock_agent:
            # Mock successful agent creation
            mock_agent.return_value = Mock()
            
            executor = GitWorkExplainerExecutor(self.test_repo_path)
            
            # Verify executor was created
            self.assertIsNotNone(executor)
            self.assertEqual(str(executor.repo_path), str(Path(self.test_repo_path).absolute()))
            
            # Verify agent was initialized
            mock_agent.assert_called_once()
    
    @unittest.skipUnless(VIJIL_EXECUTOR_AVAILABLE, "vijil_executor not available")
    def test_input_adapter(self):
        """Test the input adapter functionality."""
        with patch('vijil_executor.WorkExplainerAgent') as mock_agent:
            mock_agent.return_value = Mock()
            
            executor = GitWorkExplainerExecutor(self.test_repo_path)
            
            # Create a mock ChatCompletionRequest
            mock_request = Mock()
            mock_request.messages = [
                {"role": "user", "content": "Analyze 3 commits for executive audience"}
            ]
            
            # Test input adaptation
            result = executor.input_adapter(mock_request)
            
            # Verify the result structure
            self.assertIsInstance(result, dict)
            self.assertIn('commit_count', result)
            self.assertIn('audience', result)
            self.assertIn('repo_path', result)
            self.assertIn('interactive', result)
            
            # Verify parsing worked
            self.assertEqual(result['commit_count'], 3)
            self.assertEqual(result['interactive'], False)
    
    @unittest.skipUnless(VIJIL_EXECUTOR_AVAILABLE, "vijil_executor not available")
    def test_output_adapter(self):
        """Test the output adapter functionality."""
        with patch('vijil_executor.WorkExplainerAgent') as mock_agent:
            mock_agent.return_value = Mock()
            
            executor = GitWorkExplainerExecutor(self.test_repo_path)
            
            # Create mock agent output
            mock_summary = Mock()
            mock_summary.title = "Test Summary"
            mock_summary.audience = AudienceType.PRODUCT_MANAGER
            mock_summary.executive_summary = "Test executive summary"
            mock_summary.technical_overview = "Test technical overview"
            mock_summary.business_impact = "Test business impact"
            mock_summary.key_changes = ["Change 1", "Change 2"]
            mock_summary.next_steps = "Test next steps"
            mock_summary.work_categories = []
            
            mock_output = Mock()
            mock_output.summary = mock_summary
            
            # Test output adaptation
            result = executor.output_adapter(mock_output)
            
            # Verify result structure
            self.assertIsNotNone(result)
            self.assertEqual(result.model, "git-work-explainer")
            self.assertEqual(len(result.choices), 1)
            self.assertEqual(result.choices[0].message.role, "assistant")
            self.assertIn("Test Summary", result.choices[0].message.content)
    
    def test_parse_user_message(self):
        """Test user message parsing."""
        if not VIJIL_EXECUTOR_AVAILABLE:
            self.skipTest("vijil_executor not available")
            
        with patch('vijil_executor.WorkExplainerAgent') as mock_agent:
            mock_agent.return_value = Mock()
            
            executor = GitWorkExplainerExecutor(self.test_repo_path)
            
            # Test commit count parsing
            result = executor._parse_user_message("Analyze 5 commits for product manager")
            self.assertEqual(result.get('commit_count'), 5)
            
            # Test audience parsing
            result = executor._parse_user_message("Create summary for executive team")
            self.assertEqual(result.get('audience'), AudienceType.EXECUTIVE)
            
            # Test marketing audience
            result = executor._parse_user_message("Generate marketing report")
            self.assertEqual(result.get('audience'), AudienceType.MARKETING)
    
    def test_environment_checks(self):
        """Test environment variable checking functions."""
        if not VIJIL_EXECUTOR_AVAILABLE:
            self.skipTest("vijil_executor not available")
        
        # Test with missing environment variables
        with patch('os.getenv', return_value=None):
            self.assertFalse(check_vijil_api_key())
            self.assertFalse(check_ngrok_auth())
        
        # Test with present environment variables
        with patch('os.getenv', return_value="test_key"):
            self.assertTrue(check_vijil_api_key())
            self.assertTrue(check_ngrok_auth())
    
    @unittest.skipUnless(VIJIL_EXECUTOR_AVAILABLE, "vijil_executor not available")
    def test_vijil_evaluator_init_without_keys(self):
        """Test VijilEvaluator initialization fails without API keys."""
        with patch('os.getenv', return_value=None):
            # Mock the agent initialization to avoid LLM client requirement
            with patch('vijil_executor.WorkExplainerAgent') as mock_agent:
                mock_agent.return_value = Mock()
                with self.assertRaises(ValueError):
                    VijilEvaluator(self.test_repo_path)
    
    def test_agent_can_be_imported(self):
        """Test that the core agent can be imported and initialized."""
        try:
            # This test requires LLM API keys, so we mock the client
            with patch('core.agent.get_available_llm_client') as mock_get_client:
                mock_client = Mock()
                mock_client.__class__.__name__ = "MockClient"
                mock_get_client.return_value = mock_client
                
                agent = WorkExplainerAgent()
                self.assertIsNotNone(agent)
                
        except Exception as e:
            self.fail(f"Could not initialize WorkExplainerAgent: {e}")


class TestVijilExecutorCLI(unittest.TestCase):
    """Test CLI functionality."""
    
    @unittest.skipUnless(VIJIL_EXECUTOR_AVAILABLE, "vijil_executor not available")
    def test_cli_help_works(self):
        """Test that CLI help can be displayed."""
        import subprocess
        import sys
        
        try:
            result = subprocess.run(
                [sys.executable, 'vijil_executor.py', '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Should exit with code 0 and contain help text
            self.assertEqual(result.returncode, 0)
            self.assertIn('usage:', result.stdout.lower())
        except subprocess.TimeoutExpired:
            self.fail("CLI help command timed out")
        except FileNotFoundError:
            self.skipTest("vijil_executor.py not found")


def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Running Vijil integration tests...\n")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")
    
    return success


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)
