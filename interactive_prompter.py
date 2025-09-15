"""
Interactive prompting functionality to gather user context.
"""

from typing import Dict, Any, List, Optional
from core.models import AudienceType, CommitInfo, UserContext


class InteractivePrompter:
    """Handles interactive prompts to gather context from the developer."""
    
    def get_user_context_for_audience(self, audience: AudienceType, commits: List[CommitInfo]) -> UserContext:
        """Gather contextual information from the user about their recent work."""
        print("\n" + "="*50)
        print("WORK CONTEXT QUESTIONS")
        print("="*50)
        print(f"To generate a better {audience.value.replace('_', ' ')} summary, please answer a few questions:")
        print("(Press Enter to skip any question)\n")
        
        # Show commit preview first
        self._show_commits_preview(commits)
        
        # Ask context questions tailored to audience
        project_goal = self._ask_question(
            "What is the main goal or purpose of this recent work?"
        )
        
        business_impact = self._ask_question(
            self._get_business_impact_question(audience)
        )
        
        technical_challenges = self._ask_question(
            "Were there any significant technical challenges or decisions?"
        )
        
        next_steps = self._ask_question(
            "What are the planned next steps or follow-up work?"
        )
        
        urgency_level = self._ask_multiple_choice(
            "How would you characterize this work?",
            {
                '1': 'Critical/urgent fixes',
                '2': 'Planned feature development',
                '3': 'Maintenance/improvements',
                '4': 'Experimental/research',
                '5': 'Mixed'
            },
            default='2'
        )
        
        # Ask if there's anything else important
        additional_context = self._ask_question(
            "Anything else important to highlight in the summary?"
        )
        
        return UserContext(
            project_goal=project_goal or None,
            business_impact=business_impact or None,
            technical_challenges=technical_challenges or None,
            next_steps=next_steps or None,
            urgency_level=urgency_level,
            additional_context=additional_context or None,
            target_audience=audience
        )
    
    def _show_commits_preview(self, commits: List[CommitInfo]):
        """Show a brief preview of the commits to help user provide context."""
        print("📊 COMMITS PREVIEW:")
        print(f"   • {len(commits)} commits analyzed")
        
        if commits:
            print(f"   • Date range: {commits[-1].date.strftime('%Y-%m-%d')} to {commits[0].date.strftime('%Y-%m-%d')}")
            
            # Show recent commit messages
            print("   • Recent commits:")
            for commit in commits[:3]:
                print(f"     - {commit.sha}: {commit.short_message}")
            
            if len(commits) > 3:
                print(f"     - ... and {len(commits) - 3} more")
        
        print()
    
    def _get_business_impact_question(self, audience: AudienceType) -> str:
        """Get audience-specific business impact question."""
        questions = {
            AudienceType.PRODUCT_MANAGER: "How does this work impact product goals, user experience, or feature delivery?",
            AudienceType.EXECUTIVE: "What strategic business value does this work provide?",
            AudienceType.ENGINEERING_LEADERSHIP: "How does this work improve technical capabilities or team productivity?",
            AudienceType.MARKETING: "What customer-facing benefits or competitive advantages does this work create?",
            AudienceType.CLIENT_STAKEHOLDER: "How does this work deliver value to clients or improve their experience?",
            AudienceType.TECHNICAL_TEAM: "How does this work benefit the technical implementation or code quality?"
        }
        return questions.get(audience, "How does this work benefit users or the business?")
    
    def _ask_question(self, question: str, multiline: bool = False) -> str:
        """Ask a single question and return the response."""
        print(f"❓ {question}")
        
        if multiline:
            print("   (Type your answer. Press Ctrl+D when finished, or Enter twice for single line)")
            lines = []
            try:
                while True:
                    line = input("   ")
                    if line == "" and lines and lines[-1] == "":
                        break
                    lines.append(line)
            except EOFError:
                pass
            return "\n".join(lines).strip()
        else:
            response = input("   > ").strip()
            return response
    
    def _ask_multiple_choice(self, question: str, choices: Dict[str, str], default: str = None) -> str:
        """Ask a multiple choice question."""
        print(f"❓ {question}")
        
        for key, value in choices.items():
            marker = " (default)" if key == default else ""
            print(f"   {key}. {value}{marker}")
        
        while True:
            response = input("   > ").strip()
            
            if response == "" and default:
                return choices[default]
            elif response in choices:
                return choices[response]
            else:
                print(f"   Please enter one of: {', '.join(choices.keys())}")
    
    def _ask_yes_no(self, question: str, default: bool = None) -> bool:
        """Ask a yes/no question."""
        suffix = " (y/n)" 
        if default is True:
            suffix = " (Y/n)"
        elif default is False:
            suffix = " (y/N)"
        
        while True:
            response = input(f"❓ {question}{suffix}: ").strip().lower()
            
            if response == "" and default is not None:
                return default
            elif response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("   Please enter 'y' for yes or 'n' for no.")
