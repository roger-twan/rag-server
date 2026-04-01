"""
GitHub loaders for notes repo (real-time) and all other repos (batch).
"""

import base64
import hashlib
import hmac
import json
import logging
import re

import httpx
from llama_index.core import Document
from llama_index.readers.github import GithubClient, GithubRepositoryReader

from app.core.config import settings

# Initialize GitHub client
github_client = GithubClient(github_token=settings.GITHUB_TOKEN)

# Initialize logger
logger = logging.getLogger(__name__)

# Owner for all repos
GITHUB_OWNER = "roger-twan"


class NotesRepoLoader:
    """Loader for notes repo with specific directories and files."""

    DIRECTORIES = ["Portfolio", "Technical"]
    FILES = ["Skills.md"]
    REPO_NAME = "notes"

    @staticmethod
    def load_documents() -> list[Document]:
        """Load documents from notes repo."""
        reader = GithubRepositoryReader(
            github_client=github_client,
            owner=GITHUB_OWNER,
            repo=NotesRepoLoader.REPO_NAME,
            filter_directories=(
                NotesRepoLoader.DIRECTORIES,
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
            filter_file_paths=(
                NotesRepoLoader.FILES,
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
        )

        documents = reader.load_data(branch="main")

        # Add metadata
        for doc in documents:
            doc.metadata.update(
                {
                    "source": "github_notes",
                    "repo": NotesRepoLoader.REPO_NAME,
                    "owner": GITHUB_OWNER,
                }
            )

        return documents


class AllReposLoader:
    """Loader for all other repos (description, README, package.json)."""

    EXCLUDED_REPO = "notes"

    @staticmethod
    async def fetch_repo_list() -> list[dict]:
        """Fetch list of all repos for the owner."""
        all_repos = []
        page = 1
        per_page = 100  # Max 100 per page

        async with httpx.AsyncClient() as client:
            while True:
                response = await client.get(
                    f"https://api.github.com/users/{GITHUB_OWNER}/repos",
                    params={"per_page": per_page, "page": page},
                    headers={
                        "Authorization": f"token {settings.GITHUB_TOKEN}",
                        "Accept": "application/vnd.github.v3+json",
                    },
                )
                response.raise_for_status()
                repos = response.json()
                all_repos.extend(repos)

                # If we got less than per_page repos, we've reached the end
                if len(repos) < per_page:
                    break
                page += 1

        # Filter out the notes repo
        return [repo for repo in all_repos if repo["name"] != AllReposLoader.EXCLUDED_REPO]

    @staticmethod
    async def fetch_file_content(repo: str, path: str) -> str | None:
        """Fetch file content from a repo."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{repo}/contents/{path}",
                headers={
                    "Authorization": f"token {settings.GITHUB_TOKEN}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and "content" in data:
                # Decode base64 content
                content = base64.b64decode(data["content"]).decode("utf-8")
                return content

            return None

    @staticmethod
    async def load_repo_document(repo: dict) -> Document | None:
        """Load document for a single repo (description + README + package.json)."""
        repo_name = repo["name"]
        description = repo.get("description") or ""

        sections = []

        # Add description
        if description:
            sections.append(f"# Repository: {repo_name}\n\n**Description:** {description}\n")

        # Try to fetch README
        readme_content = await AllReposLoader.fetch_file_content(repo_name, "README.md")

        if readme_content:
            sections.append(f"\n## README\n\n{readme_content}")

        # Try to fetch package.json
        package_json = await AllReposLoader.fetch_file_content(repo_name, "package.json")
        if package_json:
            try:
                pkg_data = json.loads(package_json)
                deps = pkg_data.get("dependencies", {})
                dev_deps = pkg_data.get("devDependencies", {})

                dep_sections = []
                if deps:
                    dep_sections.append(f"**Dependencies:** {', '.join(deps.keys())}")
                if dev_deps:
                    dep_sections.append(f"**Dev Dependencies:** {', '.join(dev_deps.keys())}")

                if dep_sections:
                    sections.append(f"\n## Package.json\n\n{''.join(dep_sections)}")
            except json.JSONDecodeError:
                pass  # Skip invalid package.json

        # Try to fetch pom.xml (Maven Java)
        pom_xml = await AllReposLoader.fetch_file_content(repo_name, "pom.xml")
        if pom_xml:
            deps = AllReposLoader._parse_pom_xml(pom_xml)
            if deps:
                sections.append(
                    f"\n## Maven Dependencies (pom.xml)\n\n**Dependencies:** {', '.join(deps)}"
                )

        # Try to fetch build.gradle (Gradle Java)
        build_gradle = await AllReposLoader.fetch_file_content(repo_name, "build.gradle")
        if build_gradle:
            deps = AllReposLoader._parse_build_gradle(build_gradle)
            if deps:
                sections.append(
                    f"\n## Gradle Dependencies (build.gradle)\n\n**Dependencies:** {', '.join(deps)}"
                )

        # Try to fetch requirements.txt (Python)
        requirements = await AllReposLoader.fetch_file_content(repo_name, "requirements.txt")
        if requirements:
            deps = AllReposLoader._parse_requirements_txt(requirements)
            if deps:
                sections.append(
                    f"\n## Python Dependencies (requirements.txt)\n\n**Dependencies:** {', '.join(deps)}"
                )

        # Try to fetch pyproject.toml (Python)
        pyproject = await AllReposLoader.fetch_file_content(repo_name, "pyproject.toml")
        if pyproject:
            deps = AllReposLoader._parse_pyproject_toml(pyproject)
            if deps:
                sections.append(
                    f"\n## Python Dependencies (pyproject.toml)\n\n**Dependencies:** {', '.join(deps)}"
                )

        # Try to fetch setup.py (Python)
        setup_py = await AllReposLoader.fetch_file_content(repo_name, "setup.py")
        if setup_py:
            deps = AllReposLoader._parse_setup_py(setup_py)
            if deps:
                sections.append(
                    f"\n## Python Dependencies (setup.py)\n\n**Dependencies:** {', '.join(deps)}"
                )

        # Try to fetch Pipfile (Python)
        pipfile = await AllReposLoader.fetch_file_content(repo_name, "Pipfile")
        if pipfile:
            deps = AllReposLoader._parse_pipfile(pipfile)
            if deps:
                sections.append(
                    f"\n## Python Dependencies (Pipfile)\n\n**Dependencies:** {', '.join(deps)}"
                )

        # Try to fetch pubspec.yaml (Flutter)
        pubspec = await AllReposLoader.fetch_file_content(repo_name, "pubspec.yaml")
        if pubspec:
            deps = AllReposLoader._parse_pubspec_yaml(pubspec)
            if deps:
                sections.append(
                    f"\n## Flutter Dependencies (pubspec.yaml)\n\n**Dependencies:** {', '.join(deps)}"
                )

        if not sections:
            return None

        # Combine all sections
        full_content = "\n".join(sections)

        # Create document
        language = repo.get("language")
        document = Document(
            text=full_content,
            metadata={
                "source": "github_repos",
                "repo": repo_name,
                "owner": GITHUB_OWNER,
                "url": repo.get("html_url", ""),
                "language": language if language is not None else "",
                "stars": repo.get("stargazers_count", 0) or 0,
            },
        )

        return document

    @staticmethod
    def _parse_pom_xml(content: str) -> list[str]:
        """Parse Maven pom.xml and extract dependencies."""
        deps = []
        # Match <dependency> blocks
        pattern = r"<dependency>.*?<artifactId>(.*?)</artifactId>.*?</dependency>"
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            # Clean up the artifact ID
            artifact = match.strip()
            if artifact:
                deps.append(artifact)
        return deps

    @staticmethod
    def _parse_build_gradle(content: str) -> list[str]:
        """Parse Gradle build.gradle and extract dependencies."""
        deps = []
        # Match implementation/compile/api/testImplementation etc. dependencies
        # Pattern captures only the group ID (first part before colon)
        patterns = [
            r"(?:test)?[Ii]mplementation\s+[\'\"]([^:]+):",
            r"(?:test)?[Cc]ompile\s+[\'\"]([^:]+):",
            r"(?:test)?[Aa]pi\s+[\'\"]([^:]+):",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, content)
            deps.extend(matches)
        return deps

    @staticmethod
    def _parse_requirements_txt(content: str) -> list[str]:
        """Parse Python requirements.txt and extract package names."""
        deps = []
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Match package name before version specifier
            match = re.match(r"^([a-zA-Z0-9_-]+)", line)
            if match:
                deps.append(match.group(1))
        return deps

    @staticmethod
    def _parse_pyproject_toml(content: str) -> list[str]:
        """Parse Python pyproject.toml and extract dependencies."""
        deps = []
        # Match dependencies in [tool.poetry.dependencies] or [project] with dependencies key
        in_deps_section = False
        section_type = None
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("[tool.poetry.dependencies]"):
                in_deps_section = True
                section_type = "poetry"
                continue
            if line == "[project]":
                in_deps_section = True
                section_type = "project"
                continue
            if in_deps_section:
                if line.startswith("["):
                    break
                if section_type == "poetry":
                    match = re.match(r'^([a-zA-Z0-9_-]+)\s*[=<>"\',]', line)
                    if match and match.group(1) != "python":
                        deps.append(match.group(1))
                elif section_type == "project":
                    # Handle array format: "flask>=2.0.0",
                    match = re.match(r'^["\']([a-zA-Z0-9_-]+)', line)
                    if match:
                        deps.append(match.group(1))
        return deps

    @staticmethod
    def _parse_setup_py(content: str) -> list[str]:
        """Parse Python setup.py and extract install_requires packages."""
        deps = []
        # Match install_requires list
        pattern = r"install_requires\s*=\s*\[(.*?)\]"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            reqs_str = match.group(1)
            # Extract package names from strings
            pkg_pattern = r"['\"]([a-zA-Z0-9_-]+)"
            deps = re.findall(pkg_pattern, reqs_str)
        return deps

    @staticmethod
    def _parse_pipfile(content: str) -> list[str]:
        """Parse Python Pipfile and extract packages."""
        deps = []
        in_packages = False
        for line in content.split("\n"):
            line = line.strip()
            if line == "[packages]":
                in_packages = True
                continue
            if in_packages:
                if line.startswith("["):
                    break
                match = re.match(r"^([a-zA-Z0-9_-]+)\s*=", line)
                if match:
                    deps.append(match.group(1))
        return deps

    @staticmethod
    def _parse_pubspec_yaml(content: str) -> list[str]:
        """Parse Flutter pubspec.yaml and extract dependencies."""
        deps = []
        in_deps = False
        current_indent = 0
        last_package_name = None
        for line in content.split("\n"):
            # Skip empty lines but track position
            if not line.strip():
                continue

            # Calculate indent
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            # Check for section start
            if stripped == "dependencies:":
                in_deps = True
                current_indent = indent
                last_package_name = None
                continue
            if stripped == "dev_dependencies:":
                in_deps = True
                current_indent = indent
                last_package_name = None
                continue

            if in_deps:
                # Check if we've moved to another top-level section
                if ":" in line and not stripped.startswith("-"):
                    key = stripped.split(":")[0].strip()
                    # If same indent as dependencies section, it's a new section
                    if indent <= current_indent and key not in [
                        "dependencies",
                        "dev_dependencies",
                    ]:
                        in_deps = False
                        continue

                # Skip comments
                if stripped.startswith("#"):
                    continue

                # Check if this is a package entry at dependencies section level + 2
                if indent == current_indent + 2 and ":" in stripped:
                    # This is a new package entry
                    match = re.match(r"^([a-zA-Z0-9_-]+):", stripped)
                    if match:
                        pkg_name = match.group(1)
                        last_package_name = pkg_name
                        # Skip flutter package
                        if pkg_name == "flutter":
                            continue
                elif indent > current_indent + 2 and last_package_name:
                    # This is a nested property under a package
                    # Check if it's sdk: flutter pattern
                    if "sdk:" in stripped and "flutter" in stripped:
                        # Remove the parent package if it was added
                        if last_package_name in deps:
                            deps.remove(last_package_name)
                        continue

                # Match dependency lines like "  http: ^0.13.0" (simple value, not nested)
                match = re.match(r"^([a-zA-Z0-9_-]+):\s*[^\s:]", stripped)
                if match and match.group(1) != "flutter":
                    pkg_name = match.group(1)
                    if pkg_name not in deps:
                        deps.append(pkg_name)
        return deps

    @staticmethod
    async def load_all_documents() -> list[Document]:
        """Load documents for all repos except notes."""
        repos = await AllReposLoader.fetch_repo_list()
        total_repos = len(repos)
        documents = []

        logger.info(f"Starting to load {total_repos} GitHub repos...")

        for index, repo in enumerate(repos, 1):
            repo_name = repo.get("name", "unknown")
            try:
                logger.info(f"[{index}/{total_repos}] Loading repo: {repo_name}...")
                doc = await AllReposLoader.load_repo_document(repo)
                if doc:
                    documents.append(doc)
                    logger.info(f"[{index}/{total_repos}] Loaded repo: {repo_name}")
                else:
                    logger.info(f"[{index}/{total_repos}] Skipped repo: {repo_name} (no content)")
            except Exception as e:
                logger.warning(
                    f"[{index}/{total_repos}] Failed to load repo {repo_name}: {type(e).__name__}: {e}"
                )
                continue

        logger.info(
            f"Completed loading repos: {len(documents)}/{total_repos} repos loaded successfully"
        )
        return documents


def verify_github_webhook(payload: bytes, signature: str) -> bool:
    """
    Verify GitHub webhook signature using HMAC.

    Args:
        payload: Raw request body bytes
        signature: X-Hub-Signature-256 header value (sha256=...)

    Returns:
        True if signature is valid
    """
    if not settings.GITHUB_WEBHOOK_SECRET:
        return False

    # Compute expected signature
    secret = settings.GITHUB_WEBHOOK_SECRET.encode()
    expected_signature = "sha256=" + hmac.new(secret, payload, hashlib.sha256).hexdigest()

    # Compare signatures
    return hmac.compare_digest(expected_signature, signature)


def is_notes_repo_push(payload: dict) -> bool:
    """Check if webhook payload is for notes repo push to main branch."""
    repo_name = payload.get("repository", {}).get("name")
    ref = payload.get("ref")

    return repo_name == "notes" and ref == "refs/heads/main"
