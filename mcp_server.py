"""
MCP Server wrapper for the AI Git Work Explainer.

This module provides MCP server functionality that can be easily plugged into
AI systems like Claude Desktop, Cline, or other MCP-compatible tools.
"""

import json
import asyncio
from typing import Dict, Any, Optional

from core.agent import explain_git_work, get_repo_info


# MCP Server Tool Definitions
MCP_TOOLS = [
    {
        "name": "explain_git_work",
        "description": "Analyze recent git commits and generate business-friendly summaries for different audiences",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to git repository (default: current directory)",
                    "default": "."
                },
                "audience": {
                    "type": "string",
                    "description": "Target audience type",
                    "enum": ["product_manager", "executive", "engineering_leadership", "marketing", "client_stakeholder", "technical_team"],
                    "default": "product_manager"
                },
                "commit_count": {
                    "type": "integer",
                    "description": "Number of recent commits to analyze",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 50
                },
                "context": {
                    "type": "object",
                    "description": "Additional context about the work",
                    "properties": {
                        "project_goal": {
                            "type": "string",
                            "description": "Main goal or purpose of the recent work"
                        },
                        "business_impact": {
                            "type": "string", 
                            "description": "How the work benefits users or business"
                        },
                        "technical_challenges": {
                            "type": "string",
                            "description": "Technical challenges or decisions made"
                        },
                        "next_steps": {
                            "type": "string",
                            "description": "Planned next steps or follow-up work"
                        },
                        "additional_context": {
                            "type": "string",
                            "description": "Any other important context"
                        }
                    }
                }
            }
        }
    },
    {
        "name": "get_repository_info",
        "description": "Get basic information about a git repository",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to git repository (default: current directory)",
                    "default": "."
                }
            }
        }
    }
]


def handle_explain_git_work(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle the explain_git_work MCP tool call."""
    try:
        repo_path = args.get('repo_path', '.')
        audience = args.get('audience', 'product_manager')
        commit_count = args.get('commit_count', 5)
        context = args.get('context', {})
        
        # Call the core function
        result = explain_git_work(
            repo_path=repo_path,
            audience=audience,
            commit_count=commit_count,
            context=context
        )
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def handle_get_repository_info(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle the get_repository_info MCP tool call."""
    try:
        repo_path = args.get('repo_path', '.')
        result = get_repo_info(repo_path)
        
        return {
            "status": "success", 
            "data": result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# MCP Tool Handler Registry
MCP_HANDLERS = {
    "explain_git_work": handle_explain_git_work,
    "get_repository_info": handle_get_repository_info
}


class WorkExplainerMCPServer:
    """
    MCP Server implementation for the Work Explainer agent.
    
    This class can be used to integrate the work explainer functionality
    into MCP-compatible AI systems.
    """
    
    def __init__(self):
        self.tools = MCP_TOOLS
        self.handlers = MCP_HANDLERS
    
    def get_tools(self) -> list:
        """Return the list of available MCP tools."""
        return self.tools
    
    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP tool call."""
        if tool_name not in self.handlers:
            return {
                "status": "error",
                "error": f"Unknown tool: {tool_name}"
            }
        
        handler = self.handlers[tool_name]
        return handler(arguments)
    
    def format_tool_response(self, result: Dict[str, Any], format_type: str = "json") -> str:
        """Format tool response for different output formats."""
        if format_type == "json":
            return json.dumps(result, indent=2, default=str)
        
        elif format_type == "markdown" and result.get("status") == "success":
            data = result.get("data", {})
            if "summary" in data:
                summary = data["summary"]
                return f"""# {summary.get('title', 'Git Work Summary')}

## Executive Summary
{summary.get('executive_summary', 'N/A')}

## Technical Overview  
{summary.get('technical_overview', 'N/A')}

## Business Impact
{summary.get('business_impact', 'N/A')}

## Key Changes
{chr(10).join(f'- {change}' for change in summary.get('key_changes', []))}

## Next Steps
{summary.get('next_steps', 'N/A')}
"""
            elif "repository" in data:
                repo = data["repository"]
                activity = data.get("recent_activity", {})
                return f"""# Repository Information

**Name:** {repo.get('name', 'Unknown')}  
**Branch:** {repo.get('branch', 'Unknown')}  
**Language:** {repo.get('language', 'Unknown')}  
**Framework:** {repo.get('framework', 'Unknown')}  
**Type:** {repo.get('project_type', 'Unknown')}

## Recent Activity
- **Commits:** {activity.get('commit_count', 0)}
- **Latest:** {activity.get('latest_commit', {}).get('message', 'N/A')}
"""
        
        # Fallback to JSON for other cases
        return json.dumps(result, indent=2, default=str)


# Convenience functions for direct MCP integration
def create_mcp_server() -> WorkExplainerMCPServer:
    """Create and return an MCP server instance."""
    return WorkExplainerMCPServer()


def process_mcp_request(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single MCP request."""
    server = create_mcp_server()
    return server.handle_tool_call(tool_name, arguments)


# Example usage for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mcp_server.py <tool_name> [arguments_json]")
        print("\nAvailable tools:")
        for tool in MCP_TOOLS:
            print(f"  - {tool['name']}: {tool['description']}")
        sys.exit(1)
    
    tool_name = sys.argv[1]
    arguments = {}
    
    if len(sys.argv) > 2:
        try:
            arguments = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            print(f"Invalid JSON arguments: {sys.argv[2]}")
            sys.exit(1)
    
    # Process the request
    result = process_mcp_request(tool_name, arguments)
    print(json.dumps(result, indent=2, default=str))
