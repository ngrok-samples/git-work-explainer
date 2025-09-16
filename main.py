#!/usr/bin/env python3
"""
AI Git Work Explainer - A CLI tool that uses AI to generate business-friendly summaries of recent development work.
"""

import argparse
import asyncio
import sys
import json
import os
from pathlib import Path

from core.agent import WorkExplainerAgent
from core.models import AudienceType
from core.llm_client import get_available_llm_client




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
    

    
    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['openai', 'anthropic'],
        help='Preferred LLM provider (default: try both, OpenAI first)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Specific model to use (e.g., gpt-4o-mini, gpt-3.5-turbo, claude-3-sonnet-20240229)'
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
        

        
        print("\n✅ Setup check completed!")
        return
    
    try:
        # Initialize agent
        print("🤖 Initializing AI Git Work Explainer...")
        agent = WorkExplainerAgent(prefer_provider=args.llm_provider, model=args.model)
        
        # Show which provider and model is being used
        provider_name = agent.llm_client.__class__.__name__.replace('Client', '')
        model_name = getattr(agent.llm_client, 'model', 'unknown')
        print(f"   Using: {provider_name} ({model_name})")
        
        # Convert audience string to enum
        audience = get_audience_type(args.audience)
        

        
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
