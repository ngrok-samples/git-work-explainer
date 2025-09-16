# Vijil Integration for AI Git Work Explainer

This document explains how to use Vijil to evaluate the trustworthiness and security of the AI Git Work Explainer agent locally.

## Overview

The Vijil integration allows you to:
- Evaluate your AI agent locally without uploading code to external services
- Test against multiple security and ethics harnesses
- Monitor evaluation progress in real-time
- Generate comprehensive trustworthiness reports

## Quick Start

### 1. Setup Environment

Run the setup script to configure your environment:

```bash
python setup_vijil.py
```

This will:
- Install the Vijil SDK
- Check your Python environment
- Create configuration files (.env and setup_env.sh)
- Test your setup

### 2. Configure API Keys

You'll need the following API keys:

- **VIJIL_API_KEY**: Get from [Vijil Settings](https://app.vijil.ai/settings/api-keys)
- **NGROK_AUTHTOKEN**: Get from [Ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)
- **OPENAI_API_KEY** or **ANTHROPIC_API_KEY**: Required for the agent to work

Set them in your `.env` file or export them:

```bash
export VIJIL_API_KEY="your_vijil_api_key"
export NGROK_AUTHTOKEN="your_ngrok_auth_token"
export OPENAI_API_KEY="your_openai_api_key"
```

### 3. Check Setup

Verify everything is configured correctly:

```bash
python vijil_executor.py --check-setup
```

### 4. Run Evaluation

Start a basic security evaluation:

```bash
python vijil_executor.py
```

## Usage Examples

### Basic Evaluation

Run a standard security evaluation:

```bash
python vijil_executor.py
```

### Multiple Harnesses

Evaluate against multiple security and ethics dimensions:

```bash
python vijil_executor.py --harnesses security_Small ethics_Small privacy_Small
```

### Advanced Evaluation

Run with detailed monitoring and results:

```bash
python vijil_executor.py --advanced
```

### Specific Repository

Evaluate a different git repository:

```bash
python vijil_executor.py --repo-path /path/to/your/repo
```

### Custom Configuration

```bash
python vijil_executor.py \
  --agent-name "My Custom Agent" \
  --evaluation-name "Security Test v1.0" \
  --harnesses security_Small ethics_Small toxicity_Small \
  --rate-limit 20 \
  --rate-limit-interval 2
```

## Available Harnesses

| Harness | Description |
|---------|-------------|
| `security_Small` | Security-focused evaluation (default) |
| `ethics_Small` | Ethics and bias evaluation |
| `privacy_Small` | Privacy protection evaluation |
| `hallucination_Small` | Hallucination detection evaluation |
| `robustness_Small` | Robustness and reliability evaluation |
| `toxicity_Small` | Toxicity and harmful content evaluation |
| `stereotype_Small` | Stereotype and bias evaluation |
| `fairness_Small` | Fairness and discrimination evaluation |

## Command Line Options

```
python vijil_executor.py [OPTIONS]

Options:
  --repo-path PATH              Git repository path (default: current directory)
  --harnesses HARNESS [...]     Evaluation harnesses to run (default: security_Small)
  --agent-name NAME             Agent name for evaluation
  --evaluation-name NAME        Evaluation run name
  --advanced                    Run advanced evaluation with detailed monitoring
  --rate-limit N                Maximum requests per interval (default: 30)
  --rate-limit-interval N       Rate limit interval in minutes (default: 1)
  --check-setup                 Check if environment is configured correctly
  --help                        Show help message
```

## How It Works

The Vijil integration works by:

1. **Local Agent Executor**: Wraps your AI agent to be compatible with Vijil's evaluation APIs
2. **Input Adapter**: Converts Vijil's chat completion requests to your agent's input format
3. **Output Adapter**: Converts your agent's responses to Vijil's expected format
4. **Ngrok Tunnel**: Creates a secure, temporary endpoint for Vijil to communicate with your agent
5. **Evaluation Harnesses**: Runs predefined security and ethics tests against your agent

### Architecture

```
Vijil Cloud ←→ Ngrok Tunnel ←→ Local Agent Executor ←→ Git Work Explainer Agent
```

The agent never leaves your local environment - only the evaluation results are sent to Vijil for analysis.

## Evaluation Types

### Simple Evaluation (`vijil.local_agents.evaluate()`)

- Automatically handles agent registration and cleanup
- Shows live progress
- Can be cancelled with Ctrl+C
- Best for quick evaluations

### Advanced Evaluation (Manual Registration)

- Full control over the evaluation process
- Detailed monitoring and status reporting
- Manual cleanup required
- Better for debugging and custom workflows

## Troubleshooting

### Common Issues

**"Vijil SDK not installed"**
```bash
pip install vijil
```

**"NGROK_AUTHTOKEN is not set"**
- Get token from [Ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)
- Set with: `export NGROK_AUTHTOKEN=your_token`

**"VIJIL_API_KEY is not set"**
- Get API key from [Vijil Settings](https://app.vijil.ai/settings/api-keys)
- Set with: `export VIJIL_API_KEY=your_key`

**"No LLM client available"**
- Set either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- The agent requires an LLM provider to function

**"Agent initialization failed"**
- Check if you're in a git repository
- Verify your LLM API keys are valid
- Ensure the repository has commits to analyze

**"Evaluation failed to start"**
- Check your internet connection
- Verify all API keys are valid
- Try running `--check-setup` first

**"Rate limit exceeded"**
- Reduce `--rate-limit` parameter
- Increase `--rate-limit-interval` parameter

### Debug Mode

Run with verbose output to see detailed logs:

```bash
python vijil_executor.py --advanced
```

This will show:
- Agent registration details
- Server URL and status
- Real-time evaluation progress
- Detailed error messages

### Manual Testing

Test individual components:

```bash
# Test agent initialization
python -c "from vijil_executor import GitWorkExplainerExecutor; GitWorkExplainerExecutor()"

# Test Vijil connection
python -c "from vijil import Vijil; import os; Vijil(api_key=os.getenv('VIJIL_API_KEY'))"
```

## Security Considerations

- **API Keys**: Keep your API keys secure. Never commit them to version control.
- **Local Only**: Your code never leaves your local environment - only evaluation results are shared.
- **Temporary Endpoints**: Ngrok tunnels are temporary and automatically cleaned up.
- **Rate Limiting**: Built-in rate limiting prevents overwhelming your agent.

## Integration with Main Agent

The Vijil integration is completely separate from the main agent (`main.py`). You can:

- Run the main agent normally: `python main.py`
- Run Vijil evaluation separately: `python vijil_executor.py`
- Both use the same core agent logic but different interfaces

## Files Overview

- `vijil_executor.py` - Main Vijil integration and CLI
- `setup_vijil.py` - Environment setup and configuration
- `VIJIL_README.md` - This documentation
- `.env` - Environment variables (created by setup)
- `setup_env.sh` - Shell script for environment setup (created by setup)

## Support

For issues with:
- **Vijil platform**: [Vijil Documentation](https://docs.vijil.ai/)
- **Ngrok**: [Ngrok Documentation](https://ngrok.com/docs)
- **This integration**: Check the troubleshooting section above

## Example Workflows

### Development Testing

```bash
# Quick security check during development
python vijil_executor.py --harnesses security_Small

# More comprehensive evaluation before release
python vijil_executor.py --harnesses security_Small ethics_Small toxicity_Small --advanced
```

### CI/CD Integration

```bash
# Automated evaluation in CI/CD pipeline
export VIJIL_API_KEY=$VIJIL_KEY_FROM_SECRETS
export NGROK_AUTHTOKEN=$NGROK_TOKEN_FROM_SECRETS
python vijil_executor.py --check-setup && python vijil_executor.py --advanced
```

### Different Repositories

```bash
# Evaluate multiple projects
for repo in /path/to/repo1 /path/to/repo2; do
  python vijil_executor.py --repo-path "$repo" --evaluation-name "Evaluation for $(basename $repo)"
done
```
