"""Tests for github_loader dependency parsers."""

import base64
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.loaders.github_loader import AllReposLoader


class TestGithubHttp:
    """Tests for GitHub HTTP helper behavior."""

    @pytest.mark.asyncio
    async def test_github_get_json_retries_timeout(self):
        request = httpx.Request("GET", "https://api.github.com/test")
        response = httpx.Response(200, json={"ok": True}, request=request)
        client = AsyncMock()
        client.get.side_effect = [httpx.ConnectTimeout("timeout"), response]

        with patch("app.loaders.github_loader.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await AllReposLoader._github_get_json(client, "https://api.github.com/test")

        assert result == {"ok": True}
        assert client.get.await_count == 2
        sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_file_content_returns_none_for_404(self):
        request = httpx.Request("GET", "https://api.github.com/test")
        response = httpx.Response(404, json={"message": "Not Found"}, request=request)

        with patch(
            "app.loaders.github_loader.AllReposLoader._github_get_json",
            side_effect=httpx.HTTPStatusError("not found", request=request, response=response),
        ):
            result = await AllReposLoader.fetch_file_content(
                "repo", "missing.md", client=AsyncMock()
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_file_content_decodes_base64_content(self):
        encoded = base64.b64encode(b"# README").decode("ascii")

        with patch(
            "app.loaders.github_loader.AllReposLoader._github_get_json",
            return_value={"content": encoded},
        ):
            result = await AllReposLoader.fetch_file_content(
                "repo", "README.md", client=AsyncMock()
            )

        assert result == "# README"


class TestParsePomXml:
    """Tests for Maven pom.xml parser."""

    def test_extracts_artifact_ids(self):
        """Test that artifactIds are extracted from dependencies."""
        pom_content = """<?xml version="1.0"?>
<project>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
            <version>2.7.0</version>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>
    </dependencies>
</project>"""
        result = AllReposLoader._parse_pom_xml(pom_content)
        assert "spring-boot-starter-web" in result
        assert "spring-boot-starter-data-jpa" in result
        assert len(result) == 2

    def test_handles_empty_pom(self):
        """Test that empty pom returns empty list."""
        result = AllReposLoader._parse_pom_xml("<project></project>")
        assert result == []

    def test_handles_no_dependencies(self):
        """Test that pom without dependencies returns empty list."""
        pom = "<project><name>Test</name></project>"
        result = AllReposLoader._parse_pom_xml(pom)
        assert result == []


class TestParseBuildGradle:
    """Tests for Gradle build.gradle parser."""

    def test_extracts_implementation_deps(self):
        """Test that implementation dependencies are extracted."""
        gradle = """
plugins {
    id 'java'
}

dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web:2.7.0'
    implementation 'com.google.guava:guava:31.1-jre'
    testImplementation 'org.junit.jupiter:junit-jupiter:5.8.2'
}
"""
        result = AllReposLoader._parse_build_gradle(gradle)
        assert "org.springframework.boot" in result
        assert "com.google.guava" in result

    def test_extracts_compile_deps(self):
        """Test that compile dependencies are extracted."""
        gradle = "compile 'org.apache.commons:commons-lang3:3.12.0'"
        result = AllReposLoader._parse_build_gradle(gradle)
        assert "org.apache.commons" in result

    def test_extracts_api_deps(self):
        """Test that api dependencies are extracted."""
        gradle = "api 'org.projectlombok:lombok:1.18.24'"
        result = AllReposLoader._parse_build_gradle(gradle)
        assert "org.projectlombok" in result

    def test_handles_empty_gradle(self):
        """Test that empty build.gradle returns empty list."""
        result = AllReposLoader._parse_build_gradle("")
        assert result == []


class TestParseRequirementsTxt:
    """Tests for Python requirements.txt parser."""

    def test_extracts_package_names(self):
        """Test that package names are extracted."""
        req = """flask==2.0.1
requests>=2.25.0
numpy
pandas==1.3.0
# This is a comment
pytest~=6.2.0"""
        result = AllReposLoader._parse_requirements_txt(req)
        assert "flask" in result
        assert "requests" in result
        assert "numpy" in result
        assert "pandas" in result
        assert "pytest" in result
        assert len(result) == 5

    def test_ignores_comments(self):
        """Test that comment lines are ignored."""
        req = "# flask\nrequests"
        result = AllReposLoader._parse_requirements_txt(req)
        assert "flask" not in result
        assert "requests" in result

    def test_ignores_empty_lines(self):
        """Test that empty lines are ignored."""
        req = "flask\n\n\nrequests"
        result = AllReposLoader._parse_requirements_txt(req)
        assert len(result) == 2

    def test_handles_extras(self):
        """Test that extras are stripped."""
        req = "requests[security]>=2.25.0"
        result = AllReposLoader._parse_requirements_txt(req)
        assert "requests" in result


class TestParsePyprojectToml:
    """Tests for Python pyproject.toml parser."""

    def test_extracts_poetry_deps(self):
        """Test that Poetry dependencies are extracted."""
        toml = """[tool.poetry.dependencies]
python = "^3.9"
flask = "^2.0.0"
requests = ">=2.25.0"
numpy = "^1.21.0"

[tool.poetry.dev-dependencies]
pytest = "^6.2.0"
"""
        result = AllReposLoader._parse_pyproject_toml(toml)
        assert "flask" in result
        assert "requests" in result
        assert "numpy" in result
        assert "pytest" not in result  # Should be in dev section
        assert "python" not in result  # Should be excluded

    def test_extracts_project_deps(self):
        """Test that [project.dependencies] format is extracted."""
        toml = """[project]
name = "my-project"
dependencies = [
    "flask>=2.0.0",
    "requests>=2.25.0",
]

[project.optional-dependencies]
dev = ["pytest"]"""
        result = AllReposLoader._parse_pyproject_toml(toml)
        assert "flask" in result
        assert "requests" in result

    def test_stops_at_next_section(self):
        """Test that parsing stops at next section."""
        toml = """[tool.poetry.dependencies]
flask = "^2.0.0"

[tool.poetry.dev-dependencies]
pytest = "^6.2.0"
"""
        result = AllReposLoader._parse_pyproject_toml(toml)
        assert "flask" in result
        assert "pytest" not in result


class TestParseSetupPy:
    """Tests for Python setup.py parser."""

    def test_extracts_install_requires(self):
        """Test that install_requires packages are extracted."""
        setup = """from unittest.mock import AsyncMock, patch
setup(
    name="my-package",
    install_requires=[
        "flask>=2.0.0",
        "requests>=2.25.0",
        "numpy>=1.21.0",
    ],
)"""
        result = AllReposLoader._parse_setup_py(setup)
        assert "flask" in result
        assert "requests" in result
        assert "numpy" in result

    def test_handles_single_quotes(self):
        """Test that single-quoted packages are extracted."""
        setup = "install_requires=['flask', 'requests']"
        result = AllReposLoader._parse_setup_py(setup)
        assert "flask" in result
        assert "requests" in result

    def test_handles_empty_install_requires(self):
        """Test that empty install_requires returns empty list."""
        setup = "install_requires=[]"
        result = AllReposLoader._parse_setup_py(setup)
        assert result == []

    def test_returns_empty_when_no_install_requires(self):
        """Test that missing install_requires returns empty list."""
        setup = "setup(name='test')"
        result = AllReposLoader._parse_setup_py(setup)
        assert result == []


class TestParsePipfile:
    """Tests for Python Pipfile parser."""

    def test_extracts_packages(self):
        """Test that packages are extracted."""
        pipfile = """[[source]]
url = "https://pypi.org/simple"

[packages]
flask = "*"
requests = ">=2.25.0"
numpy = "{version=\">=1.21.0\"}"

[dev-packages]
pytest = "*"
"""
        result = AllReposLoader._parse_pipfile(pipfile)
        assert "flask" in result
        assert "requests" in result
        assert "numpy" in result
        assert "pytest" not in result

    def test_stops_at_dev_packages(self):
        """Test that parsing stops at dev-packages section."""
        pipfile = """[packages]
flask = "*"

[dev-packages]
pytest = "*"
"""
        result = AllReposLoader._parse_pipfile(pipfile)
        assert "flask" in result
        assert "pytest" not in result


class TestParsePubspecYaml:
    """Tests for Flutter pubspec.yaml parser."""

    def test_extracts_dependencies(self):
        """Test that dependencies are extracted."""
        pubspec = """name: my_app
dependencies:
  flutter:
    sdk: flutter
  http: ^0.13.0
  provider: ^6.0.0
  shared_preferences: ^2.0.0"""
        result = AllReposLoader._parse_pubspec_yaml(pubspec)
        assert "http" in result
        assert "provider" in result
        assert "shared_preferences" in result
        assert "flutter" not in result  # Should be excluded

    def test_extracts_dev_dependencies(self):
        """Test that dev_dependencies are extracted."""
        pubspec = """dependencies:
  http: ^0.13.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  mockito: ^5.0.0"""
        result = AllReposLoader._parse_pubspec_yaml(pubspec)
        assert "http" in result
        assert "mockito" in result
        assert "flutter_test" not in result

    def test_ignores_comments(self):
        """Test that comments are ignored."""
        pubspec = """dependencies:
  http: ^0.13.0
  # provider: ^6.0.0
  dio: ^4.0.0"""
        result = AllReposLoader._parse_pubspec_yaml(pubspec)
        assert "http" in result
        assert "provider" not in result
        assert "dio" in result

    def test_stops_at_next_section(self):
        """Test that parsing stops at next top-level section."""
        pubspec = """dependencies:
  http: ^0.13.0

flutter:
  uses-material-design: true"""
        result = AllReposLoader._parse_pubspec_yaml(pubspec)
        assert "http" in result
        assert "flutter" not in result
        assert "uses-material-design" not in result
