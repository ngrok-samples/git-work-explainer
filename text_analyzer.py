"""
Text analysis functionality for commit messages and file changes.
"""

import re
from typing import List, Dict, Any, Set
from collections import Counter


class TextAnalyzer:
    """Analyzes commit messages and file changes to extract meaningful patterns."""
    
    def __init__(self):
        # Common patterns in commit messages that indicate different types of work
        self.work_patterns = {
            'bug_fix': [
                r'\bfix(?:ed|es)?\b',
                r'\bbug\b',
                r'\berror\b',
                r'\bissue\b',
                r'\bproblem\b',
                r'\bresolv(?:e|ed|es)?\b'
            ],
            'feature': [
                r'\badd(?:ed|s)?\b',
                r'\bnew\b',
                r'\bfeature\b',
                r'\bimplement(?:ed|s)?\b',
                r'\bcreate(?:d|s)?\b',
                r'\bbuild\b'
            ],
            'refactor': [
                r'\brefactor(?:ed|ing)?\b',
                r'\bclean(?:ed|up)?\b',
                r'\bimprove(?:d|s|ment)?\b',
                r'\boptimiz(?:e|ed|ation)?\b',
                r'\brestructur(?:e|ed|ing)?\b'
            ],
            'update': [
                r'\bupdat(?:e|ed|ing)?\b',
                r'\bmodif(?:y|ied|ication)?\b',
                r'\bchange(?:d|s)?\b',
                r'\badjust(?:ed|ment)?\b'
            ],
            'documentation': [
                r'\bdoc(?:s|umentation)?\b',
                r'\breadme\b',
                r'\bcomment(?:s|ed)?\b',
                r'\bexample(?:s)?\b'
            ],
            'testing': [
                r'\btest(?:s|ing|ed)?\b',
                r'\bspec(?:s)?\b',
                r'\bcoverage\b',
                r'\bmock(?:s|ed|ing)?\b'
            ]
        }
        
        # File extensions that indicate different types of work
        self.file_type_patterns = {
            'frontend': ['.js', '.jsx', '.ts', '.tsx', '.vue', '.svelte', '.css', '.scss', '.html'],
            'backend': ['.py', '.java', '.go', '.rb', '.php', '.cs', '.cpp', '.c'],
            'database': ['.sql', '.migration', '.schema'],
            'config': ['.json', '.yaml', '.yml', '.toml', '.ini', '.env', '.config'],
            'documentation': ['.md', '.rst', '.txt', '.doc'],
            'testing': ['.test.js', '.test.py', '.spec.js', '_test.go', 'test_*.py']
        }
    
    def analyze_commits(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a list of commits and extract meaningful patterns."""
        analysis = {
            'total_commits': len(commits),
            'work_types': self._analyze_work_types(commits),
            'affected_areas': self._analyze_affected_areas(commits),
            'commit_frequency': self._analyze_commit_frequency(commits),
            'key_themes': self._extract_key_themes(commits),
            'technical_summary': self._generate_technical_summary(commits),
            'commits_detail': commits
        }
        
        return analysis
    
    def _analyze_work_types(self, commits: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize commits by type of work."""
        work_types = Counter()
        
        for commit in commits:
            message = commit['message'].lower()
            classified = False
            
            for work_type, patterns in self.work_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, message, re.IGNORECASE):
                        work_types[work_type] += 1
                        classified = True
                        break
                if classified:
                    break
            
            if not classified:
                work_types['other'] += 1
        
        return dict(work_types)
    
    def _analyze_affected_areas(self, commits: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze which parts of the codebase were affected."""
        affected_areas = Counter()
        
        for commit in commits:
            for file_change in commit['changed_files']:
                file_path = file_change['path'].lower()
                classified = False
                
                for area, extensions in self.file_type_patterns.items():
                    for ext in extensions:
                        if file_path.endswith(ext.lower()) or ext.lower() in file_path:
                            affected_areas[area] += 1
                            classified = True
                            break
                    if classified:
                        break
                
                if not classified:
                    # Try to infer from directory structure
                    if 'src' in file_path or 'lib' in file_path:
                        affected_areas['source_code'] += 1
                    elif 'test' in file_path:
                        affected_areas['testing'] += 1
                    elif 'doc' in file_path:
                        affected_areas['documentation'] += 1
                    else:
                        affected_areas['other'] += 1
        
        return dict(affected_areas)
    
    def _analyze_commit_frequency(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze commit frequency patterns."""
        if not commits:
            return {}
        
        dates = [commit['date'] for commit in commits]
        date_range = (max(dates) - min(dates)).days + 1
        
        return {
            'total_days': date_range,
            'commits_per_day': round(len(commits) / max(date_range, 1), 2),
            'first_commit': min(dates),
            'last_commit': max(dates)
        }
    
    def _extract_key_themes(self, commits: List[Dict[str, Any]]) -> List[str]:
        """Extract key themes from commit messages."""
        # Simple approach: find common meaningful words
        all_words = []
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        for commit in commits:
            # Extract words from commit messages
            words = re.findall(r'\b[a-zA-Z]+\b', commit['message'].lower())
            # Filter out stop words and short words
            meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
            all_words.extend(meaningful_words)
        
        # Get most common themes
        word_counts = Counter(all_words)
        return [word for word, count in word_counts.most_common(5) if count > 1]
    
    def _generate_technical_summary(self, commits: List[Dict[str, Any]]) -> str:
        """Generate a brief technical summary."""
        total_files = sum(len(commit['changed_files']) for commit in commits)
        unique_files = set()
        
        for commit in commits:
            for file_change in commit['changed_files']:
                unique_files.add(file_change['path'])
        
        return f"{len(commits)} commits affecting {len(unique_files)} unique files ({total_files} total file changes)"
