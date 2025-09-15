"""
Summary generation functionality.
"""

from typing import Dict, Any, List
from datetime import datetime


class SummaryGenerator:
    """Generates business-friendly summaries of development work."""
    
    def generate_summary(self, analysis: Dict[str, Any], context: Dict[str, Any], format_type: str = 'markdown') -> str:
        """Generate a summary based on analysis and user context."""
        
        if format_type == 'markdown':
            return self._generate_markdown_summary(analysis, context)
        else:
            return self._generate_text_summary(analysis, context)
    
    def _generate_markdown_summary(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a markdown-formatted summary."""
        sections = []
        
        # Header
        sections.append(self._generate_header(analysis, context))
        
        # Executive Summary
        sections.append(self._generate_executive_summary(analysis, context))
        
        # What Was Built
        sections.append(self._generate_what_was_built(analysis, context))
        
        # Technical Overview
        sections.append(self._generate_technical_overview(analysis, context))
        
        # Business Impact
        sections.append(self._generate_business_impact(analysis, context))
        
        # Next Steps
        sections.append(self._generate_next_steps(analysis, context))
        
        # Technical Details (Optional)
        if self._should_include_technical_details(context):
            sections.append(self._generate_technical_details(analysis))
        
        return "\n\n".join(sections)
    
    def _generate_text_summary(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a plain text summary."""
        # Similar structure but without markdown formatting
        markdown_summary = self._generate_markdown_summary(analysis, context)
        
        # Strip markdown formatting for plain text
        import re
        text_summary = re.sub(r'#+\s*', '', markdown_summary)  # Remove headers
        text_summary = re.sub(r'\*\*(.*?)\*\*', r'\1', text_summary)  # Remove bold
        text_summary = re.sub(r'\*(.*?)\*', r'\1', text_summary)  # Remove italic
        text_summary = re.sub(r'`(.*?)`', r'\1', text_summary)  # Remove code formatting
        text_summary = re.sub(r'^\s*[-*]\s*', '• ', text_summary, flags=re.MULTILINE)  # Convert bullets
        
        return text_summary
    
    def _generate_header(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate the document header."""
        date_range = self._get_date_range(analysis)
        
        return f"""# Development Work Summary
        
**Period:** {date_range}  
**Commits Analyzed:** {analysis['total_commits']}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
    
    def _generate_executive_summary(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        summary_parts = []
        
        # Start with user's project goal if provided
        if context.get('project_goal'):
            summary_parts.append(f"**Objective:** {context['project_goal']}")
        
        # Add technical summary
        summary_parts.append(f"**Scope:** {analysis['technical_summary']}")
        
        # Add work type summary
        if analysis['work_types']:
            primary_work = self._get_primary_work_type(analysis['work_types'])
            summary_parts.append(f"**Primary Focus:** {primary_work}")
        
        # Add urgency context if provided
        if context.get('urgency_level') and context['urgency_level'] != 'Mixed':
            summary_parts.append(f"**Nature:** {context['urgency_level']}")
        
        summary_text = "\n".join(f"- {part}" for part in summary_parts)
        
        return f"""## Executive Summary

{summary_text}"""
    
    def _generate_what_was_built(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate 'what was built' section."""
        content = []
        
        # Use context if available, otherwise infer from analysis
        if context.get('project_goal'):
            content.append(context['project_goal'])
        
        # Add details from analysis
        if analysis['work_types']:
            work_summary = self._summarize_work_types(analysis['work_types'])
            content.append(f"This involved {work_summary}.")
        
        if analysis['affected_areas']:
            areas_summary = self._summarize_affected_areas(analysis['affected_areas'])
            content.append(f"Changes were made primarily to {areas_summary}.")
        
        if analysis['key_themes']:
            themes = ', '.join(analysis['key_themes'][:3])
            content.append(f"Key focus areas included: {themes}.")
        
        content_text = " ".join(content) if content else "Development work was completed across multiple areas of the codebase."
        
        return f"""## What Was Built

{content_text}"""
    
    def _generate_technical_overview(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate technical overview section."""
        overview_parts = []
        
        # Commit summary
        frequency = analysis.get('commit_frequency', {})
        if frequency.get('commits_per_day'):
            overview_parts.append(f"**Development Pace:** {frequency['commits_per_day']} commits per day over {frequency.get('total_days', 'N/A')} days")
        
        # Work breakdown
        if analysis['work_types']:
            work_breakdown = []
            for work_type, count in analysis['work_types'].items():
                percentage = round((count / analysis['total_commits']) * 100)
                work_breakdown.append(f"{work_type.replace('_', ' ').title()}: {count} commits ({percentage}%)")
            
            overview_parts.append(f"**Work Breakdown:**\n" + "\n".join(f"- {item}" for item in work_breakdown))
        
        # Technical challenges
        if context.get('technical_challenges'):
            overview_parts.append(f"**Technical Challenges:** {context['technical_challenges']}")
        
        overview_text = "\n\n".join(overview_parts)
        
        return f"""## Technical Overview

{overview_text}"""
    
    def _generate_business_impact(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate business impact section."""
        impact_content = []
        
        if context.get('business_impact'):
            impact_content.append(context['business_impact'])
        else:
            # Try to infer business impact from work types
            impact_content.append(self._infer_business_impact(analysis['work_types']))
        
        impact_text = " ".join(impact_content)
        
        return f"""## Business Impact

{impact_text}"""
    
    def _generate_next_steps(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate next steps section."""
        if context.get('next_steps'):
            return f"""## Next Steps

{context['next_steps']}"""
        else:
            return f"""## Next Steps

Follow-up work and next priorities to be determined."""
    
    def _generate_technical_details(self, analysis: Dict[str, Any]) -> str:
        """Generate detailed technical appendix."""
        details = []
        
        # Recent commits
        details.append("### Recent Commits")
        for commit in analysis['commits_detail'][:5]:  # Show last 5
            commit_date = commit['date'].strftime('%Y-%m-%d')
            details.append(f"- **{commit['sha']}** ({commit_date}): {commit['message']}")
        
        # File changes
        if analysis.get('affected_areas'):
            details.append("\n### Areas Modified")
            for area, count in analysis['affected_areas'].items():
                details.append(f"- {area.replace('_', ' ').title()}: {count} files")
        
        return f"""## Technical Details

{chr(10).join(details)}"""
    
    def _should_include_technical_details(self, context: Dict[str, Any]) -> bool:
        """Determine if technical details should be included."""
        audience = context.get('target_audience', '')
        return 'engineering' in audience.lower() or 'technical' in audience.lower()
    
    def _get_date_range(self, analysis: Dict[str, Any]) -> str:
        """Get formatted date range from analysis."""
        frequency = analysis.get('commit_frequency', {})
        if frequency.get('first_commit') and frequency.get('last_commit'):
            start = frequency['first_commit'].strftime('%Y-%m-%d')
            end = frequency['last_commit'].strftime('%Y-%m-%d')
            if start == end:
                return start
            return f"{start} to {end}"
        return "Recent commits"
    
    def _get_primary_work_type(self, work_types: Dict[str, int]) -> str:
        """Get the primary work type with nice formatting."""
        if not work_types:
            return "Development work"
        
        primary = max(work_types.items(), key=lambda x: x[1])
        return primary[0].replace('_', ' ').title()
    
    def _summarize_work_types(self, work_types: Dict[str, int]) -> str:
        """Summarize work types in natural language."""
        if not work_types:
            return "various development tasks"
        
        sorted_types = sorted(work_types.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_types) == 1:
            return f"{sorted_types[0][0].replace('_', ' ')}"
        elif len(sorted_types) == 2:
            return f"{sorted_types[0][0].replace('_', ' ')} and {sorted_types[1][0].replace('_', ' ')}"
        else:
            primary = sorted_types[0][0].replace('_', ' ')
            return f"primarily {primary}, with additional work on {sorted_types[1][0].replace('_', ' ')}"
    
    def _summarize_affected_areas(self, affected_areas: Dict[str, int]) -> str:
        """Summarize affected areas in natural language."""
        if not affected_areas:
            return "various parts of the system"
        
        sorted_areas = sorted(affected_areas.items(), key=lambda x: x[1], reverse=True)
        primary_area = sorted_areas[0][0].replace('_', ' ')
        
        if len(sorted_areas) == 1:
            return f"the {primary_area} components"
        else:
            return f"the {primary_area} components and other areas"
    
    def _infer_business_impact(self, work_types: Dict[str, int]) -> str:
        """Infer business impact from work types."""
        if not work_types:
            return "This work contributes to the overall improvement and maintenance of the system."
        
        primary_type = max(work_types.items(), key=lambda x: x[1])[0]
        
        impact_map = {
            'feature': "This work adds new capabilities that enhance user experience and expand product functionality.",
            'bug_fix': "This work improves system reliability and user experience by resolving issues.",
            'refactor': "This work improves code quality and maintainability, supporting long-term product development.",
            'update': "This work ensures the system stays current and maintains optimal performance.",
            'documentation': "This work improves team productivity and knowledge sharing.",
            'testing': "This work increases system reliability and reduces risk of future issues."
        }
        
        return impact_map.get(primary_type, "This work contributes to the overall improvement and maintenance of the system.")
