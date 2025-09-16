#!/usr/bin/env python3
"""
Setup script for Vijil integration with AI Git Work Explainer

This script helps users configure their environment for Vijil evaluations.
"""

import os
import sys
from pathlib import Path
import subprocess


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True


def check_virtual_environment():
    """Check if we're in a virtual environment."""
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    if in_venv:
        print(f"✅ Virtual environment detected: {sys.prefix}")
    else:
        print("⚠️  Not in a virtual environment. Consider using venv for isolation.")
    
    return in_venv


def install_vijil():
    """Install Vijil SDK."""
    try:
        import vijil
        print("✅ Vijil SDK is already installed")
        return True
    except ImportError:
        print("📦 Installing Vijil SDK...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'vijil'])
            print("✅ Vijil SDK installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Vijil SDK: {e}")
            return False


def check_environment_variables():
    """Check and help set up environment variables."""
    print("\n🔧 Checking environment variables...")
    
    # Check VIJIL_API_KEY
    vijil_key = os.getenv("VIJIL_API_KEY")
    if vijil_key:
        print("✅ VIJIL_API_KEY is set")
    else:
        print("❌ VIJIL_API_KEY is not set")
        print("   Get your API key from: https://app.vijil.ai/settings/api-keys")
        print("   Then set it with: export VIJIL_API_KEY=your_api_key_here")
    
    # Check NGROK_AUTHTOKEN
    ngrok_token = os.getenv("NGROK_AUTHTOKEN")
    if ngrok_token:
        print("✅ NGROK_AUTHTOKEN is set")
    else:
        print("❌ NGROK_AUTHTOKEN is not set")
        print("   Get your auth token from: https://dashboard.ngrok.com/get-started/your-authtoken")
        print("   Then set it with: export NGROK_AUTHTOKEN=your_auth_token_here")
    
    # Check LLM API keys (required for the agent to work)
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key:
        print("✅ OPENAI_API_KEY is set")
    elif anthropic_key:
        print("✅ ANTHROPIC_API_KEY is set")
    else:
        print("❌ Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set")
        print("   The agent requires at least one LLM provider API key")
        print("   Set with: export OPENAI_API_KEY=your_key or export ANTHROPIC_API_KEY=your_key")
    
    return vijil_key and ngrok_token and (openai_key or anthropic_key)


def create_env_file():
    """Create a .env file template."""
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists - not overwriting")
        return
    
    env_template = """# Vijil API Configuration
# Get your API key from: https://app.vijil.ai/settings/api-keys
VIJIL_API_KEY=your_vijil_api_key_here

# Ngrok Configuration (required for local agent evaluation)  
# Get your auth token from: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTHTOKEN=your_ngrok_auth_token_here

# LLM Provider API Keys (at least one required)
# OpenAI API Key (get from: https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (get from: https://console.anthropic.com/settings/keys)  
ANTHROPIC_API_KEY=your_anthropic_api_key_here
"""
    
    env_file.write_text(env_template)
    print(f"✅ Created .env template file: {env_file.absolute()}")
    print("   Edit this file with your actual API keys")


def create_shell_script():
    """Create a shell script to set environment variables."""
    script_content = """#!/bin/bash
# Vijil Environment Setup Script
# Source this file to set environment variables: source setup_env.sh

# Vijil API Configuration
export VIJIL_API_KEY="your_vijil_api_key_here"

# Ngrok Configuration  
export NGROK_AUTHTOKEN="your_ngrok_auth_token_here"

# LLM Provider API Keys (at least one required)
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

echo "Environment variables set for Vijil evaluation"
echo "Remember to replace the placeholder values with your actual API keys"
"""
    
    script_path = Path("setup_env.sh")
    script_path.write_text(script_content)
    script_path.chmod(0o755)  # Make executable
    print(f"✅ Created shell script: {script_path.absolute()}")
    print("   Edit and source with: source setup_env.sh")


def test_setup():
    """Test if the setup is working."""
    print("\n🧪 Testing setup...")
    
    try:
        # Test Vijil import
        import vijil
        print("✅ Vijil SDK can be imported")
        
        # Test agent initialization
        from vijil_executor import GitWorkExplainerExecutor
        executor = GitWorkExplainerExecutor()
        print("✅ Git Work Explainer executor can be initialized")
        
        # Test environment variables
        all_vars_set = check_environment_variables()
        if all_vars_set:
            print("✅ All required environment variables are set")
        else:
            print("⚠️  Some environment variables are missing")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions."""
    print("""
📖 USAGE INSTRUCTIONS:

1. Basic Evaluation:
   python vijil_executor.py

2. Check Setup:
   python vijil_executor.py --check-setup

3. Advanced Evaluation:
   python vijil_executor.py --advanced

4. Custom Evaluation:
   python vijil_executor.py --harnesses security_Small ethics_Small

5. Specific Repository:
   python vijil_executor.py --repo-path /path/to/your/repo

6. Help:
   python vijil_executor.py --help

🔧 ENVIRONMENT SETUP:

Option 1: Use .env file (recommended)
   - Edit the .env file created by this script
   - The vijil_executor.py will automatically load it

Option 2: Use shell script
   - Edit setup_env.sh created by this script
   - Run: source setup_env.sh

Option 3: Set manually
   export VIJIL_API_KEY=your_actual_key
   export NGROK_AUTHTOKEN=your_actual_token
   export OPENAI_API_KEY=your_actual_key

🚀 QUICK START:

1. Get API keys:
   - Vijil: https://app.vijil.ai/settings/api-keys
   - Ngrok: https://dashboard.ngrok.com/get-started/your-authtoken
   - OpenAI: https://platform.openai.com/api-keys

2. Set environment variables (use .env file or export commands)

3. Run evaluation:
   python vijil_executor.py --check-setup
   python vijil_executor.py

🔍 TROUBLESHOOTING:

- If you get "command not found", make sure you're in the virtual environment
- If Vijil import fails, try: pip install vijil
- If ngrok fails, ensure NGROK_AUTHTOKEN is set correctly
- For LLM errors, verify your OPENAI_API_KEY or ANTHROPIC_API_KEY
    """)


def main():
    """Main setup function."""
    print("🚀 Vijil Integration Setup for AI Git Work Explainer\n")
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    check_virtual_environment()
    
    # Install dependencies
    if not install_vijil():
        print("❌ Setup failed - could not install Vijil SDK")
        sys.exit(1)
    
    # Check environment
    env_ok = check_environment_variables()
    
    # Create helper files
    print("\n📁 Creating configuration files...")
    create_env_file()
    create_shell_script()
    
    # Test setup
    if test_setup():
        print("\n✅ Setup completed successfully!")
    else:
        print("\n⚠️  Setup completed with warnings")
    
    # Print instructions
    print_usage_instructions()
    
    if not env_ok:
        print("\n⚠️  NEXT STEPS:")
        print("1. Edit .env file or setup_env.sh with your actual API keys")
        print("2. Run: python vijil_executor.py --check-setup")
        print("3. Run: python vijil_executor.py")


if __name__ == '__main__':
    main()
