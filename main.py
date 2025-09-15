#!/usr/bin/env python3
"""
AI Git Work Explainer - A CLI tool that uses AI to generate business-friendly summaries of recent development work.
"""

import argparse
import asyncio
import sys
import json
from pathlib import Path

from core.agent import WorkExplainerAgent
from core.models import AudienceType
from core.llm_client import get_available_llm_client
from core.vijil_client import VijilEvaluateClient
from core.evaluation_harnesses import EvaluationHarness, run_quick_evaluation, run_scenario_test


def get_audience_type(audience_str: str) -> AudienceType:
    """Convert string to AudienceType enum."""
    audience_map = {
        'pm': AudienceType.PRODUCT_MANAGER,
        'product_manager': AudienceType.PRODUCT_MANAGER,
        'executive': AudienceType.EXECUTIVE,
        'exec': AudienceType.EXECUTIVE,
        'engineering': AudienceType.ENGINEERING_LEADERSHIP,
        'eng_leadership': AudienceType.ENGINEERING_LEADERSHIP,
        'marketing': AudienceType.MARKETING,
        'client': AudienceType.CLIENT_STAKEHOLDER,
        'stakeholder': AudienceType.CLIENT_STAKEHOLDER,
        'technical': AudienceType.TECHNICAL_TEAM,
        'tech_team': AudienceType.TECHNICAL_TEAM
    }
    return audience_map.get(audience_str.lower(), AudienceType.PRODUCT_MANAGER)


def format_output(response, format_type: str) -> str:
    """Format the agent response for output."""
    summary = response.summary
    
    if format_type == 'json':
        return json.dumps({
            "title": summary.title,
            "executive_summary": summary.executive_summary,
            "technical_overview": summary.technical_overview,
            "business_impact": summary.business_impact,
            "key_changes": summary.key_changes,
            "next_steps": summary.next_steps,
            "work_categories": [cat.value for cat in summary.work_categories],
            "audience": summary.audience.value,
            "metadata": {
                "processing_time": response.processing_time,
                "confidence_score": response.confidence_score,
                "tokens_used": response.tokens_used
            }
        }, indent=2)
    
    elif format_type == 'markdown':
        return f"""# {summary.title}

**Target Audience:** {summary.audience.value.replace('_', ' ').title()}  
**Generated:** AI Analysis ({response.processing_time:.1f}s, {response.tokens_used} tokens)

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

---
*Work Categories: {', '.join(cat.value.replace('_', ' ').title() for cat in summary.work_categories)}*
"""
    
    else:  # text format
        return f"""{summary.title}

Target Audience: {summary.audience.value.replace('_', ' ').title()}
Generated: AI Analysis ({response.processing_time:.1f}s, {response.tokens_used} tokens)

EXECUTIVE SUMMARY
{summary.executive_summary}

TECHNICAL OVERVIEW
{summary.technical_overview}

BUSINESS IMPACT
{summary.business_impact}

KEY CHANGES
{chr(10).join(f'• {change}' for change in summary.key_changes)}

NEXT STEPS
{summary.next_steps}

Work Categories: {', '.join(cat.value.replace('_', ' ').title() for cat in summary.work_categories)}
"""


async def run_full_evaluation_suite(agent: WorkExplainerAgent, args):
    """Run the complete Vijil evaluation suite."""
    print("🧪 Running full evaluation suite...")
    
    try:
        vijil_client = VijilEvaluateClient()
        harness = EvaluationHarness(agent, vijil_client)
        
        results = await harness.run_full_evaluation_suite()
        
        # Format and save results
        if args.evaluation_report or args.output:
            output_file = args.output or "evaluation_report.json"
            output_path = Path(output_file)
            output_path.write_text(json.dumps(results, indent=2, default=str))
            print(f"📊 Evaluation report saved to: {output_path}")
        
        # Print summary
        overall_metrics = results.get("overall_metrics", {})
        print(f"\n🎯 EVALUATION RESULTS")
        print(f"   Trust Score: {overall_metrics.get('average_trust_score', 0):.2f}/1.0")
        print(f"   Consistency Score: {overall_metrics.get('average_consistency_score', 0):.2f}/1.0")
        print(f"   Pass Rate: {overall_metrics.get('pass_rate', 0):.1%}")
        
        recommendations = results.get("recommendations", [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations[:3]:
                print(f"   • {rec}")
    
    except Exception as e:
        print(f"❌ Evaluation suite failed: {e}")
        if "VIJIL_API_KEY" in str(e):
            print("💡 Tip: Set VIJIL_API_KEY environment variable to enable Vijil integration")


async def run_evaluation_scenario(agent: WorkExplainerAgent, scenario: str, args):
    """Run a specific evaluation scenario."""
    print(f"🎯 Running evaluation scenario: {scenario}")
    
    try:
        result = await run_scenario_test(agent, scenario)
        
        print(f"\n📊 SCENARIO RESULTS: {scenario}")
        print(f"   Trust Score: {result.trust_score:.2f}/1.0")
        print(f"   Issues Found: {len(result.issues_found)}")
        
        # Show trust dimensions
        print(f"\n🔍 TRUST DIMENSIONS:")
        for dimension, score in result.dimensions.items():
            status = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
            print(f"   {status} {dimension.replace('_', ' ').title()}: {score:.2f}")
        
        # Show recommendations
        if result.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in result.recommendations:
                print(f"   • {rec}")
        
        # Save detailed results if requested
        if args.evaluation_report or args.output:
            output_file = args.output or f"evaluation_{scenario}.json"
            output_path = Path(output_file)
            result_dict = {
                "evaluation_id": result.evaluation_id,
                "scenario": result.test_scenario,
                "trust_score": result.trust_score,
                "dimensions": result.dimensions,
                "issues_found": result.issues_found,
                "recommendations": result.recommendations,
                "timestamp": result.timestamp.isoformat()
            }
            output_path.write_text(json.dumps(result_dict, indent=2))
            print(f"📊 Detailed results saved to: {output_path}")
    
    except Exception as e:
        print(f"❌ Evaluation scenario failed: {e}")


async def run_quick_evaluation_mode(agent: WorkExplainerAgent, args):
    """Run quick evaluation mode."""
    print("⚡ Running quick evaluation...")
    
    try:
        results = await run_quick_evaluation(agent)
        
        # Show basic results
        if "basic_feature" in results:
            result = results["basic_feature"]
            print(f"\n📊 QUICK EVALUATION RESULTS")
            print(f"   Trust Score: {result.trust_score:.2f}/1.0")
            print(f"   Issues Found: {len(result.issues_found)}")
            
            if result.trust_score >= 0.7:
                print("   ✅ Agent performance is within acceptable range")
            else:
                print("   ⚠️  Agent performance may need improvement")
        
        if "consistency" in results:
            consistency = results["consistency"]
            print(f"   Consistency Score: {consistency.consistency_score:.2f}/1.0")
    
    except Exception as e:
        print(f"❌ Quick evaluation failed: {e}")


async def run_consistency_test(agent: WorkExplainerAgent, args):
    """Run consistency testing."""
    print(f"🔄 Testing consistency ({args.consistency_runs} runs)...")
    
    try:
        vijil_client = VijilEvaluateClient()
        harness = EvaluationHarness(agent, vijil_client)
        
        result = await harness.test_consistency("basic_feature_development", args.consistency_runs)
        
        print(f"\n🔄 CONSISTENCY TEST RESULTS")
        print(f"   Consistency Score: {result.consistency_score:.2f}/1.0")
        print(f"   Identical Outputs: {result.identical_outputs}/{result.runs}")
        print(f"   Similar Outputs: {result.similar_outputs}/{result.runs}")
        print(f"   Different Outputs: {result.different_outputs}/{result.runs}")
        
        # Show variance metrics
        if result.variance_metrics:
            print(f"\n📊 VARIANCE METRICS:")
            for metric, value in result.variance_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   • {metric.replace('_', ' ').title()}: {value:.3f}")
        
        if result.consistency_score >= 0.8:
            print("   ✅ Agent shows good consistency")
        else:
            print("   ⚠️  Agent consistency may need improvement")
    
    except Exception as e:
        print(f"❌ Consistency test failed: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="AI-powered tool to generate business-friendly summaries of recent development work",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive analysis for product managers
  %(prog)s -n 10 --audience executive         # Executive summary of 10 commits
  %(prog)s -n 5 --output report.md --no-interactive  # Save summary without prompts
  %(prog)s --audience marketing --format json # Marketing summary in JSON format
  %(prog)s --llm-provider anthropic           # Use Claude instead of GPT
  %(prog)s --evaluate --llm-provider openai   # Test GPT with Vijil evaluation

Audience Types:
  pm, product_manager    - Product managers (default)
  executive, exec        - Executive team
  engineering, eng       - Engineering leadership
  marketing              - Marketing team
  client, stakeholder    - Client stakeholders
  technical, tech        - Technical team

LLM Providers:
  openai                 - OpenAI GPT models (default)
  anthropic             - Anthropic Claude models
        """
    )
    
    parser.add_argument(
        '-n', '--commits',
        type=int,
        default=5,
        help='Number of recent commits to analyze (default: 5)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path (default: stdout)'
    )
    
    parser.add_argument(
        '--repo-path',
        type=str,
        default='.',
        help='Path to git repository (default: current directory)'
    )
    
    parser.add_argument(
        '--audience',
        type=str,
        default='product_manager',
        help='Target audience (default: product_manager)'
    )
    
    parser.add_argument(
        '--format',
        choices=['markdown', 'text', 'json'],
        default='markdown',
        help='Output format (default: markdown)'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Skip interactive prompts'
    )
    
    parser.add_argument(
        '--check-setup',
        action='store_true',
        help='Check if LLM APIs are properly configured'
    )
    
    # Vijil Evaluate integration options
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run Vijil trustworthiness evaluation of the agent'
    )
    
    parser.add_argument(
        '--evaluation-scenario',
        type=str,
        choices=['basic_feature_development', 'bug_fix_analysis', 'large_refactor', 'audience_adaptation'],
        help='Run a specific evaluation scenario'
    )
    
    parser.add_argument(
        '--test-consistency',
        action='store_true',
        help='Test agent consistency across multiple runs'
    )
    
    parser.add_argument(
        '--consistency-runs',
        type=int,
        default=3,
        help='Number of runs for consistency testing (default: 3)'
    )
    
    parser.add_argument(
        '--evaluation-report',
        action='store_true',
        help='Generate detailed evaluation report'
    )
    
    parser.add_argument(
        '--full-evaluation-suite',
        action='store_true',
        help='Run complete evaluation suite (all scenarios + consistency tests)'
    )
    
    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['openai', 'anthropic'],
        help='Preferred LLM provider (default: try both, OpenAI first)'
    )
    
    args = parser.parse_args()
    
    # Check setup if requested
    if args.check_setup:
        print("🔍 Checking AI Git Work Explainer setup...\n")
        
        # Check LLM providers
        print("🤖 LLM PROVIDERS:")
        
        # Check OpenAI
        try:
            from core.llm_client import OpenAIClient
            openai_client = OpenAIClient()
            if openai_client.is_available():
                print("   ✅ OpenAI (GPT) - Available")
            else:
                print("   ❌ OpenAI (GPT) - Missing API key")
        except ImportError:
            print("   ⚠️  OpenAI (GPT) - Package not installed")
        except Exception as e:
            print(f"   ❌ OpenAI (GPT) - Error: {e}")
        
        # Check Anthropic
        try:
            from core.llm_client import AnthropicClient
            anthropic_client = AnthropicClient()
            if anthropic_client.is_available():
                print("   ✅ Anthropic (Claude) - Available")
            else:
                print("   ❌ Anthropic (Claude) - Missing API key")
        except ImportError:
            print("   ⚠️  Anthropic (Claude) - Package not installed")
        except Exception as e:
            print(f"   ❌ Anthropic (Claude) - Error: {e}")
        
        # Check which one would be used
        llm_client = get_available_llm_client(args.llm_provider)
        if llm_client:
            print(f"\n   🎯 Will use: {llm_client.__class__.__name__}")
        else:
            print("\n   ❌ No LLM provider available!")
            print("      Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
            sys.exit(1)
        
        # Check Vijil setup
        print(f"\n🧪 EVALUATION SYSTEM:")
        try:
            vijil_client = VijilEvaluateClient()
            print(f"   ✅ Vijil Evaluate - Available")
        except Exception as e:
            print(f"   ⚠️  Vijil Evaluate - Not configured: {e}")
            print("      Set VIJIL_API_KEY environment variable for evaluation features")
            print("      (Optional - basic evaluation still works without it)")
        
        print("\n✅ Setup check completed!")
        return
    
    try:
        # Initialize agent
        print("🤖 Initializing AI Git Work Explainer...")
        agent = WorkExplainerAgent(prefer_provider=args.llm_provider)
        
        # Show which provider is being used
        provider_name = agent.llm_client.__class__.__name__.replace('Client', '')
        print(f"   Using: {provider_name}")
        
        # Convert audience string to enum
        audience = get_audience_type(args.audience)
        
        # Handle evaluation modes
        if args.full_evaluation_suite:
            await run_full_evaluation_suite(agent, args)
            return
            
        elif args.evaluation_scenario:
            await run_evaluation_scenario(agent, args.evaluation_scenario, args)
            return
            
        elif args.evaluate:
            await run_quick_evaluation_mode(agent, args)
            return
            
        elif args.test_consistency:
            await run_consistency_test(agent, args)
            return
        
        # Regular analysis mode
        response = await agent.explain_work(
            commit_count=args.commits,
            audience=audience,
            repo_path=args.repo_path,
            interactive=not args.no_interactive
        )
        
        # Format output
        output_text = format_output(response, args.format)
        
        # Save or print results
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(output_text)
            print(f"\n✅ Summary saved to: {output_path}")
        else:
            print("\n" + "="*60)
            print("🎯 WORK SUMMARY")
            print("="*60)
            print(output_text)
            
    except KeyboardInterrupt:
        print("\n\n👋 Analysis cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
