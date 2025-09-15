"""
Git repository analysis functionality - refactored for LLM integration.
"""

import git
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from core.models import CommitInfo, FileChange, RepositoryContext


class GitAnalyzer:
    """Handles git repository operations and commit analysis."""
    
    def __init__(self, repo_path: str = '.'):
        """Initialize with a git repository path."""
        try:
            self.repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            raise ValueError(f"No valid git repository found at: {repo_path}")
        except git.NoSuchPathError:
            raise ValueError(f"Path does not exist: {repo_path}")
    
    def get_recent_commits(self, count: int = 5) -> List[CommitInfo]:
        """Get the last N commits with rich metadata for LLM analysis."""
        commits = []
        
        try:
            for commit in self.repo.iter_commits(max_count=count):
                # Get changed files with detailed information
                changed_files = self._get_changed_files(commit)
                
                # Generate diff summary for LLM context
                diff_summary = self._generate_diff_summary(commit)
                
                commit_info = CommitInfo(
                    sha=commit.hexsha[:8],
                    full_sha=commit.hexsha,
                    message=commit.message.strip(),
                    author=str(commit.author),
                    author_email=commit.author.email,
                    date=datetime.fromtimestamp(commit.committed_date),
                    changed_files=changed_files,
                    diff_summary=diff_summary
                )
                commits.append(commit_info)
        except Exception as e:
            raise RuntimeError(f"Error reading git commits: {e}")
        
        return commits
    
    def _get_changed_files(self, commit) -> List[FileChange]:
        """Get detailed list of files changed in a commit."""
        changed_files = []
        
        try:
            # Handle initial commit case
            if len(commit.parents) == 0:
                for item in commit.tree.traverse():
                    if item.type == 'blob':  # It's a file
                        changed_files.append(FileChange(
                            path=item.path,
                            change_type='added',
                            additions=0,  # Can't calculate for initial commit
                            deletions=0
                        ))
            else:
                # Compare with parent commit
                parent = commit.parents[0]
                diffs = parent.diff(commit, create_patch=True)
                
                for diff in diffs:
                    # Calculate line changes
                    additions, deletions = self._calculate_line_changes(diff)
                    
                    file_change = FileChange(
                        path=diff.b_path or diff.a_path,
                        change_type=self._get_change_type(diff),
                        additions=additions,
                        deletions=deletions
                    )
                    changed_files.append(file_change)
                    
        except Exception as e:
            # If we can't get diff info, just note that files were changed
            changed_files.append(FileChange(
                path='unknown',
                change_type='modified',
                additions=0,
                deletions=0
            ))
        
        return changed_files
    
    def _get_change_type(self, diff) -> str:
        """Determine the type of change for a file."""
        if diff.new_file:
            return 'added'
        elif diff.deleted_file:
            return 'deleted'
        elif diff.renamed_file:
            return 'renamed'
        else:
            return 'modified'
    
    def _get_commit_stats(self, commit) -> Dict[str, int]:
        """Get basic statistics for a commit."""
        stats = {
            'files_changed': 0,
            'insertions': 0,
            'deletions': 0
        }
        
        try:
            if len(commit.parents) > 0:
                parent = commit.parents[0]
                diff_stats = parent.diff(commit).iter_change_type('M')
                
                for diff in diff_stats:
                    stats['files_changed'] += 1
                    # Note: Getting exact line counts requires more complex parsing
                    # For MVP, we'll keep it simple
                    
        except Exception:
            # If stats fail, just return empty stats
            pass
        
        return stats
    
    def get_repository_context(self) -> RepositoryContext:
        """Get rich repository context for LLM analysis."""
        try:
            name = Path(self.repo.working_dir).name
            branch = self.repo.active_branch.name
            remote_url = self._get_remote_url()
            
            # Detect language and framework
            language = self._detect_primary_language()
            framework = self._detect_framework()
            project_type = self._detect_project_type()
            
            return RepositoryContext(
                name=name,
                branch=branch,
                remote_url=remote_url,
                language=language,
                framework=framework,
                project_type=project_type
            )
        except Exception as e:
            return RepositoryContext(
                name='unknown',
                branch='unknown',
                remote_url=None,
                language=None,
                framework=None,
                project_type=None
            )
    
    def _get_remote_url(self) -> Optional[str]:
        """Get the remote URL if available."""
        try:
            return list(self.repo.remotes.origin.urls)[0]
        except:
            return None
    
    def _calculate_line_changes(self, diff) -> tuple[int, int]:
        """Calculate additions and deletions from a diff."""
        try:
            if hasattr(diff, 'diff') and diff.diff:
                patch = diff.diff.decode('utf-8', errors='ignore')
                additions = patch.count('\n+') - patch.count('\n+++')
                deletions = patch.count('\n-') - patch.count('\n---')
                return max(0, additions), max(0, deletions)
        except Exception:
            pass
        return 0, 0
    
    def _generate_diff_summary(self, commit) -> str:
        """Generate a brief summary of what changed in the commit."""
        try:
            if len(commit.parents) == 0:
                return "Initial commit - new repository created"
            
            parent = commit.parents[0]
            diffs = parent.diff(commit)
            
            file_count = len(diffs)
            if file_count == 0:
                return "No file changes detected"
            
            # Count change types
            added = sum(1 for d in diffs if d.new_file)
            deleted = sum(1 for d in diffs if d.deleted_file)
            modified = file_count - added - deleted
            
            summary_parts = []
            if added:
                summary_parts.append(f"{added} files added")
            if modified:
                summary_parts.append(f"{modified} files modified") 
            if deleted:
                summary_parts.append(f"{deleted} files deleted")
            
            return f"{file_count} files changed: " + ", ".join(summary_parts)
            
        except Exception:
            return "Changes detected but unable to analyze diff"
    
    def _detect_primary_language(self) -> Optional[str]:
        """Detect the primary programming language."""
        try:
            # Count file extensions in the repository
            extensions = {}
            for item in self.repo.tree().traverse():
                if item.type == 'blob':
                    path = item.path
                    if '.' in path:
                        ext = path.split('.')[-1].lower()
                        extensions[ext] = extensions.get(ext, 0) + 1
            
            # Map extensions to languages
            language_map = {
                'py': 'Python',
                'js': 'JavaScript', 
                'ts': 'TypeScript',
                'java': 'Java',
                'go': 'Go',
                'rs': 'Rust',
                'cpp': 'C++',
                'c': 'C',
                'rb': 'Ruby',
                'php': 'PHP',
                'cs': 'C#',
                'swift': 'Swift',
                'kt': 'Kotlin',
                'scala': 'Scala'
            }
            
            # Find most common language extension
            if extensions:
                most_common_ext = max(extensions.items(), key=lambda x: x[1])[0]
                return language_map.get(most_common_ext, most_common_ext.upper())
                
        except Exception:
            pass
        return None
    
    def _detect_framework(self) -> Optional[str]:
        """Detect framework or major libraries."""
        try:
            # Check for common framework files
            framework_indicators = {
                'package.json': ['React', 'Vue', 'Angular', 'Node.js'],
                'requirements.txt': ['Django', 'Flask', 'FastAPI'],
                'Gemfile': ['Rails', 'Sinatra'],
                'pom.xml': ['Spring'],
                'build.gradle': ['Spring', 'Android'],
                'Cargo.toml': ['Rust'],
                'go.mod': ['Go'],
                'composer.json': ['Laravel', 'Symfony']
            }
            
            for item in self.repo.tree().traverse():
                if item.type == 'blob':
                    filename = Path(item.path).name
                    if filename in framework_indicators:
                        # Could parse the file content to determine specific framework
                        return framework_indicators[filename][0]  # Return first option for now
                        
        except Exception:
            pass
        return None
    
    def _detect_project_type(self) -> Optional[str]:
        """Detect the type of project."""
        try:
            # Look for indicators of project type
            indicators = {
                'web_app': ['src/components', 'public/index.html', 'views/', 'templates/'],
                'api': ['api/', 'endpoints/', 'routes/', 'controllers/'],
                'library': ['lib/', 'package.json', '__init__.py'],
                'cli_tool': ['bin/', 'cmd/', 'cli.py', 'main.py'],
                'mobile_app': ['ios/', 'android/', 'App.js'],
                'data_science': ['notebooks/', 'data/', 'models/', '*.ipynb']
            }
            
            repo_files = [item.path for item in self.repo.tree().traverse() if item.type == 'blob']
            
            for project_type, patterns in indicators.items():
                for pattern in patterns:
                    if any(pattern in path for path in repo_files):
                        return project_type.replace('_', ' ').title()
                        
        except Exception:
            pass
        return None
