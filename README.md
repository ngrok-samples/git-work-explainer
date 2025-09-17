# AI Git Work Explainer 🤖

An AI-powered CLI tool that analyzes git commits and generates business-friendly summaries for non-technical stakeholders. Uses LLM APIs (OpenAI/Anthropic) to understand your development work and explain it in terms that matter to different audiences.

## ✨ Features

- **🧠 AI-Powered Analysis**: Uses GPT-4 or Claude to understand commit context and generate intelligent summaries
- **🎯 Audience-Aware**: Tailors explanations for product managers, executives, marketing, clients, etc.
- **💬 Interactive Context**: Asks smart questions to gather business context
- **🔍 Rich Git Analysis**: Extracts commit messages, file changes, diffs, and repository context
- **📊 Multiple Formats**: Outputs markdown, text, or JSON
- **🔧 Vijil Integration**: Evaluates agent performance using Vijil benchmarks
- **⚡ Async Architecture**: Fast, modern Python with async/await

## 🚀 Quick Start

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

## 📖 Usage

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

## 🏗 Architecture

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

## 📊 Example Output

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

## 🛠 Development

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

## 🧪 Vijil Evaluate Integration

This tool integrates with **[Vijil Evaluate](https://docs.vijil.ai/)** to test the trustworthiness of your AI agent. Vijil helps you understand how reliable, consistent, and safe your AI explanations are.

### Trust Evaluation Features

- **🎯 Accuracy Testing** - Validates that summaries match actual git changes
- **🔄 Consistency Testing** - Ensures similar outputs for same inputs
- **👥 Audience Appropriateness** - Checks language fits target audience
- **🔒 Safety Checks** - Detects potential sensitive information leaks
- **📊 Comprehensive Metrics** - 9 dimensions of trust evaluation

### Evaluation Usage

```bash
# Check if Vijil is configured
python main.py --check-setup

# Quick trustworthiness evaluation
python main.py --evaluate

# Test specific scenarios
python main.py --evaluation-scenario basic_feature_development

# Test consistency (multiple runs)
python main.py --test-consistency --consistency-runs 5

# Full evaluation suite
python main.py --full-evaluation-suite --evaluation-report

# Save evaluation results
python main.py --evaluate --output trust_report.json
```

### Evaluation Scenarios

- **Basic Feature Development** - Tests standard feature work analysis
- **Bug Fix Analysis** - Validates bug fix explanations
- **Large Refactor** - Tests complex refactoring summaries
- **Audience Adaptation** - Checks different audience targeting

### Trust Dimensions Measured

1. **Accuracy** - Summary matches git changes
2. **Completeness** - All important aspects covered
3. **Appropriateness** - Language fits target audience
4. **Consistency** - Similar inputs produce similar outputs
5. **Safety** - No sensitive information exposed
6. **Relevance** - Content relates to actual changes
7. **Clarity** - Summary is clear and understandable
8. **Factuality** - Claims are factual and verifiable
9. **Reliability** - Agent performs predictably

## 🔑 Environment Variables

- `OPENAI_API_KEY` - OpenAI API key for GPT models
- `ANTHROPIC_API_KEY` - Anthropic API key for Claude models  
- `VIJIL_API_KEY` - Vijil API key for trustworthiness evaluation

## 🚧 Roadmap

**Core Features:**
- [ ] Support for additional LLM providers (Gemini, local models)
- [ ] Integration with project management tools (Jira, Linear)
- [ ] Customizable summary templates
- [ ] Team collaboration features
- [ ] Web interface
- [ ] Slack/Teams bot integration

**Vijil Integration:**
- [x] Vijil Evaluate integration for trustworthiness testing
- [ ] Vijil Dome integration for real-time guardrails
- [ ] Custom evaluation metrics for domain-specific use cases
- [ ] Automated evaluation in CI/CD pipelines
- [ ] Advanced consistency testing with different LLM providers
- [ ] Integration with Vijil's red-teaming capabilities

## 🤝 Contributing

This tool is architected for easy extension:

- **New LLM providers**: Add clients in `core/llm_client.py`
- **Enhanced git analysis**: Extend `git_analyzer.py`
- **Custom output formats**: Modify formatters in `main.py`
- **Evaluation harnesses**: Add new evaluation types in `vijil_executor.py`

## 📄 License

MIT License - see LICENSE file for details.

---

*Built with ❤️ for developers who need to explain their work to humans*
