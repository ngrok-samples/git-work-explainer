# AI Git Work Explainer

An AI-powered CLI tool that analyzes git commits and generates business-friendly summaries for non-technical stakeholders. Uses LLM APIs (OpenAI/Anthropic) to understand your development work and explain it in terms that matter to different audiences.

## Features

- **AI-Powered Analysis**: Uses GPT-4 or Claude to understand commit context and generate intelligent summaries
- **Audience-Aware**: Tailors explanations for product managers, executives, marketing, clients, etc.
- **Interactive Context**: Asks smart questions to gather business context
- **Rich Git Analysis**: Extracts commit messages, file changes, diffs, and repository context
- **Multiple Formats**: Outputs markdown, text, or JSON
- **Vijil Integration**: Full evaluation suite with multiple LLM providers and performance benchmarking
- **Async Architecture**: Fast, modern Python with async/await

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your API key** (choose one or both):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
   export VIJIL_API_KEY="your-vijil-api-key-here"  # Optional for evaluation
   ```

3. **Run analysis**:
   ```bash
   python main.py
   ```

## Usage

### Basic Usage

```bash
# Interactive analysis for product managers (default)
python main.py

# Executive summary of last 10 commits
python main.py -n 10 --audience executive

# Save marketing summary without prompts
python main.py --audience marketing --no-interactive -o summary.md

# Get JSON output for integration
python main.py --format json --audience technical

# Use Claude instead of GPT
python main.py --llm-provider anthropic

# Use specific provider for evaluation
python main.py --evaluate --llm-provider openai
```

### Check Setup

```bash
# Verify API configuration for all providers
python main.py --check-setup

# Check specific provider preference
python main.py --check-setup --llm-provider anthropic
```

### Audience Types

- `pm`, `product_manager` - Product managers (default)
- `executive`, `exec` - Executive team  
- `engineering`, `eng` - Engineering leadership
- `marketing` - Marketing team
- `client`, `stakeholder` - Client stakeholders
- `technical`, `tech` - Technical team

## Vijil Evaluation

This agent integrates with [Vijil](https://vijil.ai) to evaluate and benchmark its performance on various tasks. Vijil tests the agent's ability to analyze git repositories and provide appropriate responses to different types of inputs.

### Quick Start with Vijil

1. **Get your Vijil API key** from the [Vijil dashboard](https://vijil.ai)
2. **Set up environment**:
   ```bash
   export VIJIL_API_KEY="your-vijil-api-key-here"
   export OPENAI_API_KEY="your-openai-key"  # or ANTHROPIC_API_KEY
   ```

3. **Test your API key**:
   ```bash
   python vijil_executor.py --test-api-key
   ```

4. **Run evaluation**:
   ```bash
   # Basic evaluation with OpenAI
   python vijil_executor.py --simple-name --llm-provider openai
   
   # Evaluation with Anthropic
   python vijil_executor.py --simple-name --llm-provider anthropic
   
   # Custom evaluation name
   python vijil_executor.py --evaluation-name "my-test-v1" --simple-name
   ```

### Evaluation Options

```bash
# Choose LLM provider for evaluation
python vijil_executor.py --llm-provider anthropic|openai

# Use simple agent naming (recommended)
python vijil_executor.py --simple-name

# Specify evaluation name
python vijil_executor.py --evaluation-name "my-evaluation"

# Use pre-created API key from dashboard
python vijil_executor.py --api-key-name "my-key-name"

# Select specific test harnesses
python vijil_executor.py --harnesses security reasoning

# Generate report from existing evaluation
python vijil_executor.py --generate-report EVALUATION_ID
```

### How Vijil Testing Works

When Vijil evaluates this agent:

1. **Vijil sends test prompts** (e.g., "What's the capital of France?", "Explain quantum computing")
2. **Agent analyzes actual git commits** from this repository instead of answering the prompt directly
3. **LLM receives both** the git analysis data AND Vijil's test prompt as "additional context"
4. **Agent responds** with git commit analysis, potentially influenced by the test prompt context
5. **Vijil evaluates** how well the agent handled the mixed input scenario

This tests the agent's ability to stay focused on its core task (git analysis) while handling potentially irrelevant or out-of-scope user input.

### Troubleshooting Vijil Integration

**API Key Issues:**
```bash
# Test API key validity
python vijil_executor.py --test-api-key

# Use simple naming to avoid UUID validation errors
python vijil_executor.py --simple-name
```

**LLM Provider Issues:**
```bash
# Verify your LLM API keys work
python main.py --check-setup

# Try different provider
python vijil_executor.py --llm-provider anthropic  # or openai
```

**Common Errors:**
- `'NoneType' object is not subscriptable`: Use `--simple-name` flag
- `API token validation failed`: Check your `VIJIL_API_KEY` environment variable
- `Empty response content`: LLM provider issue, try different provider or check API keys

## Architecture

This tool is designed as an **AI agent** that orchestrates between different data sources and AI APIs:

```
Git Repository → Git Analyzer → AI Agent → LLM API → Formatted Summary
                      ↑             ↓
                User Context ← Interactive Prompter
```

### Core Components

- **`core/agent.py`** - Main AI agent orchestrator
- **`core/llm_client.py`** - LLM API integrations (OpenAI, Anthropic)
- **`core/models.py`** - Data models and types
- **`git_analyzer.py`** - Rich git repository analysis
- **`interactive_prompter.py`** - Smart context gathering
- **`vijil_executor.py`** - Vijil integration for agent evaluation

## Example Output

```markdown
# User Authentication System Implementation

**Target Audience:** Product Manager  
**Generated:** AI Analysis (2.3s, 1,247 tokens)

## Executive Summary

Implemented a comprehensive user authentication system including login, registration, 
and session management. This work establishes the foundation for user account 
features and improves security posture significantly.

## Technical Overview

The implementation spans 8 commits across authentication middleware, user models, 
API endpoints, and frontend components. Key technical decisions included JWT token 
strategy, password hashing with bcrypt, and session management architecture.

## Business Impact

This work enables user personalization features, improves data security compliance, 
and provides the foundation for premium user accounts. Users can now securely 
create accounts and access personalized experiences.

## Key Changes

- Added JWT-based authentication middleware
- Created user registration and login API endpoints  
- Implemented secure password hashing and validation
- Built frontend login/register forms with error handling
- Added session management and logout functionality

## Next Steps

- Implement password reset functionality
- Add OAuth integration for social logins
- Create user profile management features
- Set up email verification system

---
*Work Categories: Feature, Security*
```

## Development

### Project Structure

```
ai-agent-vigil/
├── core/                   # Core agent functionality
│   ├── agent.py           # Main AI agent
│   ├── llm_client.py      # LLM API clients  
│   └── models.py          # Data models
├── git_analyzer.py        # Git repository analysis
├── interactive_prompter.py # User interaction
├── main.py               # CLI interface
├── vijil_executor.py     # Vijil evaluation integration
└── requirements.txt      # Dependencies
```

### Adding New LLM Providers

1. Create a new client class inheriting from `LLMClient` in `core/llm_client.py`
2. Implement the `analyze_commits` method
3. Add detection logic in `get_available_llm_client()`

### Adding New Audience Types

1. Add enum values to `AudienceType` in `core/models.py`
2. Update prompt templates in `core/llm_client.py`
3. Add CLI mappings in `main.py`



## Environment Variables

- `OPENAI_API_KEY` - OpenAI API key for GPT models
- `ANTHROPIC_API_KEY` - Anthropic API key for Claude models  
- `VIJIL_API_KEY` - Vijil API key for agent evaluation and benchmarking
- `NGROK_AUTHTOKEN` ngrok auth token to create a tunnel to the local agent
