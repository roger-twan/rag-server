"""
GitHub loaders for notes repo (real-time) and all other repos (batch).
"""

import base64
import hashlib
import hmac
import json
import re

import httpx
from llama_index.core import Document
from llama_index.readers.github import GithubClient, GithubRepositoryReader

from app.core.config import settings

# Initialize GitHub client
github_client = GithubClient(github_token=settings.GITHUB_TOKEN)

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
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.github.com/users/{GITHUB_OWNER}/repos",
                headers={
                    "Authorization": f"token {settings.GITHUB_TOKEN}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            response.raise_for_status()
            repos = response.json()

        # Filter out the notes repo
        return [repo for repo in repos if repo["name"] != AllReposLoader.EXCLUDED_REPO]

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
        if not readme_content:
            # Try without extension
            readme_content = await AllReposLoader.fetch_file_content(repo_name, "README")

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
        document = Document(
            text=full_content,
            metadata={
                "source": "github_repos",
                "repo": repo_name,
                "owner": GITHUB_OWNER,
                "url": repo.get("html_url", ""),
                "language": repo.get("language", ""),
                "stars": repo.get("stargazers_count", 0),
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
        # Match implementation/compile/api dependencies
        patterns = [
            r"implementation\s+['\"]([^:]+):([^:]+)['\"]",
            r"compile\s+['\"]([^:]+):([^:]+)['\"]",
            r"api\s+['\"]([^:]+):([^:]+)['\"]",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                deps.append(match[0])  # Group ID
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
        # Match dependencies in [tool.poetry.dependencies] or [project.dependencies]
        in_deps_section = False
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("[tool.poetry.dependencies]") or line.startswith(
                "[project.dependencies]"
            ):
                in_deps_section = True
                continue
            if in_deps_section:
                if line.startswith("["):
                    break
                match = re.match(r"^([a-zA-Z0-9_-]+)\s*=", line)
                if match and match.group(1) != "python":
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
        for line in content.split("\n"):
            line = line.strip()
            if line == "dependencies:":
                in_deps = True
                continue
            if line == "dev_dependencies:":
                in_deps = True
                continue
            if in_deps:
                if not line or line.startswith("#"):
                    continue
                # Check if we've moved to another section
                if ":" in line and not line.startswith("-"):
                    key = line.split(":")[0].strip()
                    if key not in ["dependencies", "dev_dependencies"]:
                        in_deps = False
                        continue
                match = re.match(r"^([a-zA-Z0-9_-]+):", line)
                if match and match.group(1) != "flutter":
                    deps.append(match.group(1))
        return deps

    @staticmethod
    async def load_all_documents() -> list[Document]:
        """Load documents for all repos except notes."""
        repos = await AllReposLoader.fetch_repo_list()
        documents = []

        for repo in repos:
            doc = await AllReposLoader.load_repo_document(repo)
            if doc:
                documents.append(doc)

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
