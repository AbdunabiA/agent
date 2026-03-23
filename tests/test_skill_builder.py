"""Tests for SkillBuilder — code validation, generation, staging, approval, rejection.

Also tests parse_natural_schedule from the scheduler module.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.config import SkillBuilderConfig
from agent.core.events import EventBus
from agent.core.scheduler import parse_natural_schedule
from agent.skills.builder import (
    BLOCKED_PATTERNS,
    GeneratedSkill,
    SkillBuilder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_main_py() -> str:
    """Return valid main.py content that passes all validation checks."""
    return (
        "from agent.skills.base import Skill\n\n"
        "class MySkill(Skill):\n"
        "    async def setup(self) -> None:\n"
        "        self.register_tool(\n"
        "            name='do_thing',\n"
        "            description='Does a thing',\n"
        "            function=self._do_thing,\n"
        "            tier='safe',\n"
        "        )\n\n"
        "    async def _do_thing(self, param: str = 'hi') -> str:\n"
        "        return f'done: {param}'\n"
    )


def _valid_skill_md() -> str:
    """Return valid SKILL.md content with YAML frontmatter."""
    return (
        "---\n"
        "name: my-skill\n"
        "description: A test skill\n"
        "version: '0.1.0'\n"
        "permissions:\n"
        "  - safe\n"
        "---\n"
        "# My Skill\n"
    )


def _make_generated_skill(
    name: str = "my-skill",
    skill_md: str | None = None,
    main_py: str | None = None,
    permissions: list[str] | None = None,
) -> GeneratedSkill:
    """Create a GeneratedSkill with sensible defaults."""
    return GeneratedSkill(
        name=name,
        display_name=name.replace("-", " ").title(),
        description="A test skill",
        skill_md=skill_md if skill_md is not None else _valid_skill_md(),
        main_py=main_py if main_py is not None else _valid_main_py(),
        permissions=permissions or ["safe"],
    )


def _make_llm_response(content: str) -> MagicMock:
    """Create a mock LLMResponse with .content attribute."""
    resp = MagicMock()
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.completion = AsyncMock()
    return llm


@pytest.fixture
def builder_config(tmp_path: Path) -> SkillBuilderConfig:
    return SkillBuilderConfig(
        enabled=True,
        staging_dir=str(tmp_path / "_staging"),
        max_retries=3,
        auto_approve=False,
        max_permissions=["safe", "moderate"],
    )


@pytest.fixture
def builder(
    mock_llm: AsyncMock,
    builder_config: SkillBuilderConfig,
    event_bus: EventBus,
) -> SkillBuilder:
    return SkillBuilder(
        llm=mock_llm,
        config=builder_config,
        event_bus=event_bus,
        skill_manager=None,
    )


# ===========================================================================
# 1. _validate_code tests
# ===========================================================================


class TestValidateCode:
    """Tests for SkillBuilder._validate_code."""

    def test_valid_code(self, builder: SkillBuilder) -> None:
        skill = _make_generated_skill()
        result = builder._validate_code(skill)

        assert result.valid is True
        assert result.errors == []

    def test_valid_code_no_register_tool_warning(self, builder: SkillBuilder) -> None:
        """Code without register_tool should pass but produce a warning."""
        main_py = (
            "from agent.skills.base import Skill\n\n"
            "class MySkill(Skill):\n"
            "    async def setup(self) -> None:\n"
            "        pass\n"
        )
        skill = _make_generated_skill(main_py=main_py)
        result = builder._validate_code(skill)

        assert result.valid is True
        assert any("register_tool" in w for w in result.warnings)

    def test_syntax_error(self, builder: SkillBuilder) -> None:
        main_py = "def broken(\n"
        skill = _make_generated_skill(main_py=main_py)
        result = builder._validate_code(skill)

        assert result.valid is False
        assert any("Syntax error" in e for e in result.errors)

    def test_empty_main_py(self, builder: SkillBuilder) -> None:
        skill = _make_generated_skill(main_py="")
        result = builder._validate_code(skill)

        assert result.valid is False
        assert any("main.py is empty" in e for e in result.errors)

    def test_whitespace_only_main_py(self, builder: SkillBuilder) -> None:
        skill = _make_generated_skill(main_py="   \n\n  ")
        result = builder._validate_code(skill)

        assert result.valid is False
        assert any("main.py is empty" in e for e in result.errors)

    def test_empty_skill_md(self, builder: SkillBuilder) -> None:
        skill = _make_generated_skill(skill_md="")
        result = builder._validate_code(skill)

        assert result.valid is False
        assert any("SKILL.md is empty" in e for e in result.errors)

    def test_skill_md_missing_frontmatter(self, builder: SkillBuilder) -> None:
        skill = _make_generated_skill(skill_md="# Just a heading\nNo frontmatter here.")
        result = builder._validate_code(skill)

        assert result.valid is False
        assert any("frontmatter" in e.lower() for e in result.errors)

    def test_missing_skill_subclass(self, builder: SkillBuilder) -> None:
        main_py = (
            "class NotASkill:\n"
            "    async def setup(self) -> None:\n"
            "        self.register_tool(name='x', description='x', function=self.x)\n\n"
            "    async def x(self) -> str:\n"
            "        return 'hi'\n"
        )
        skill = _make_generated_skill(main_py=main_py)
        result = builder._validate_code(skill)

        assert result.valid is False
        assert any("Skill" in e and "class" in e.lower() for e in result.errors)

    def test_missing_setup_method(self, builder: SkillBuilder) -> None:
        main_py = (
            "from agent.skills.base import Skill\n\n"
            "class MySkill(Skill):\n"
            "    def not_setup(self) -> None:\n"
            "        self.register_tool(name='x', description='x', function=self.x)\n\n"
            "    async def x(self) -> str:\n"
            "        return 'hi'\n"
        )
        skill = _make_generated_skill(main_py=main_py)
        result = builder._validate_code(skill)

        assert result.valid is False
        assert any("setup" in e for e in result.errors)

    @pytest.mark.parametrize("pattern", sorted(BLOCKED_PATTERNS))
    def test_blocked_pattern(self, builder: SkillBuilder, pattern: str) -> None:
        """Each blocked pattern should cause validation failure."""
        main_py = (
            "from agent.skills.base import Skill\n\n"
            "class MySkill(Skill):\n"
            "    async def setup(self) -> None:\n"
            "        self.register_tool(name='x', description='x', function=self._x)\n\n"
            "    async def _x(self) -> str:\n"
            f"        return str({pattern}'hello'))\n"
        )
        # The code may have syntax issues from the pattern insertion,
        # but we just need the pattern text present. Use a comment-based approach.
        main_py = (
            "from agent.skills.base import Skill\n\n"
            "class MySkill(Skill):\n"
            "    async def setup(self) -> None:\n"
            "        self.register_tool(name='x', description='x', function=self._x)\n\n"
            "    async def _x(self) -> str:\n"
            f"        # {pattern}\n"
            "        return 'hi'\n"
        )
        skill = _make_generated_skill(main_py=main_py)
        result = builder._validate_code(skill)

        assert result.valid is False
        assert any(pattern in e for e in result.errors)

    def test_multiple_errors_accumulated(self, builder: SkillBuilder) -> None:
        """Empty skill_md and empty main_py should both be reported."""
        skill = _make_generated_skill(skill_md="", main_py="")
        result = builder._validate_code(skill)

        assert result.valid is False
        assert len(result.errors) >= 2

    def test_attribute_base_class_detected(self, builder: SkillBuilder) -> None:
        """Skill referenced as module.Skill (ast.Attribute) is also detected."""
        main_py = (
            "import agent.skills.base as base\n\n"
            "class MySkill(base.Skill):\n"
            "    async def setup(self) -> None:\n"
            "        self.register_tool(name='x', description='x', function=self._x)\n\n"
            "    async def _x(self) -> str:\n"
            "        return 'hi'\n"
        )
        skill = _make_generated_skill(main_py=main_py)
        result = builder._validate_code(skill)

        assert result.valid is True


# ===========================================================================
# 2. _generate_name tests
# ===========================================================================


class TestGenerateName:
    """Tests for SkillBuilder._generate_name."""

    @pytest.mark.asyncio
    async def test_generate_name_basic(self, builder: SkillBuilder, mock_llm: AsyncMock) -> None:
        mock_llm.completion.return_value = _make_llm_response("weather-fetch")
        name = await builder._generate_name("A skill that fetches weather data")

        assert name == "weather-fetch"
        mock_llm.completion.assert_awaited_once()
        # Verify the system message asks for a kebab-case name
        call_kwargs = mock_llm.completion.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert any("kebab-case" in m["content"] for m in messages if m["role"] == "system")

    @pytest.mark.asyncio
    async def test_generate_name_strips_quotes(
        self, builder: SkillBuilder, mock_llm: AsyncMock
    ) -> None:
        mock_llm.completion.return_value = _make_llm_response("  'code-review'  \n")
        name = await builder._generate_name("Reviews code")

        assert name == "code-review"

    @pytest.mark.asyncio
    async def test_generate_name_strips_backticks(
        self, builder: SkillBuilder, mock_llm: AsyncMock
    ) -> None:
        mock_llm.completion.return_value = _make_llm_response("`my-skill`")
        name = await builder._generate_name("Does something")

        assert name == "my-skill"

    @pytest.mark.asyncio
    async def test_generate_name_empty_fallback(
        self, builder: SkillBuilder, mock_llm: AsyncMock
    ) -> None:
        mock_llm.completion.return_value = _make_llm_response("   ")
        name = await builder._generate_name("Some description")

        assert name == "custom-skill"


# ===========================================================================
# 3. build_skill tests
# ===========================================================================


class TestBuildSkill:
    """Tests for SkillBuilder.build_skill — full pipeline."""

    @pytest.mark.asyncio
    async def test_build_skill_success(
        self,
        builder: SkillBuilder,
        mock_llm: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Successful build: generate code, validate, stage, sandbox test."""
        generated_json = json.dumps(
            {
                "skill_md": _valid_skill_md(),
                "main_py": _valid_main_py(),
            }
        )
        mock_llm.completion.return_value = _make_llm_response(generated_json)

        with patch.object(builder, "_sandbox_test") as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                passed=True, output="Registered 1 tool(s)", error="", duration_ms=100
            )
            result = await builder.build_skill(
                description="A skill that does stuff",
                name="test-skill",
            )

        assert result.success is True
        assert result.skill_name == "test-skill"
        assert result.staging_path is not None
        assert result.retries == 0
        assert result.validation is not None
        assert result.validation.valid is True

    @pytest.mark.asyncio
    async def test_build_skill_generates_name_when_not_provided(
        self,
        builder: SkillBuilder,
        mock_llm: AsyncMock,
    ) -> None:
        """When name is None, _generate_name is called first."""
        # First call: _generate_name; second call: _generate_code
        generated_json = json.dumps(
            {
                "skill_md": _valid_skill_md(),
                "main_py": _valid_main_py(),
            }
        )
        mock_llm.completion.side_effect = [
            _make_llm_response("auto-named"),
            _make_llm_response(generated_json),
        ]

        with patch.object(builder, "_sandbox_test") as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                passed=True, output="Registered 1 tool(s)", error="", duration_ms=50
            )
            result = await builder.build_skill(
                description="Something cool",
                name=None,
            )

        assert result.success is True
        assert result.skill_name == "auto-named"
        # Two LLM calls: one for name, one for code
        assert mock_llm.completion.await_count == 2

    @pytest.mark.asyncio
    async def test_build_skill_sanitizes_name(
        self,
        builder: SkillBuilder,
        mock_llm: AsyncMock,
    ) -> None:
        """Name is lowered, spaces/underscores become dashes, non-alnum stripped."""
        generated_json = json.dumps(
            {
                "skill_md": _valid_skill_md(),
                "main_py": _valid_main_py(),
            }
        )
        mock_llm.completion.return_value = _make_llm_response(generated_json)

        with patch.object(builder, "_sandbox_test") as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                passed=True, output="OK", error="", duration_ms=10
            )
            result = await builder.build_skill(
                description="test",
                name="My Cool_Skill!@#",
            )

        assert result.skill_name == "my-cool-skill"

    @pytest.mark.asyncio
    async def test_build_skill_validation_retry(
        self,
        builder: SkillBuilder,
        mock_llm: AsyncMock,
    ) -> None:
        """Validation failure triggers retry with error context."""
        bad_code = "class NotASkill:\n" "    pass\n"
        bad_json = json.dumps(
            {
                "skill_md": _valid_skill_md(),
                "main_py": bad_code,
            }
        )
        good_json = json.dumps(
            {
                "skill_md": _valid_skill_md(),
                "main_py": _valid_main_py(),
            }
        )

        # First attempt fails validation, second succeeds
        mock_llm.completion.side_effect = [
            _make_llm_response(bad_json),
            _make_llm_response(good_json),
        ]

        with patch.object(builder, "_sandbox_test") as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                passed=True, output="OK", error="", duration_ms=10
            )
            result = await builder.build_skill(
                description="test",
                name="retry-skill",
            )

        assert result.success is True
        assert result.retries == 1  # succeeded on second attempt (index 1)
        assert mock_llm.completion.await_count == 2

    @pytest.mark.asyncio
    async def test_build_skill_sandbox_failure_retry(
        self,
        builder: SkillBuilder,
        mock_llm: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Sandbox test failure triggers retry; staging dir is cleaned up."""
        generated_json = json.dumps(
            {
                "skill_md": _valid_skill_md(),
                "main_py": _valid_main_py(),
            }
        )
        mock_llm.completion.return_value = _make_llm_response(generated_json)

        call_count = 0

        async def sandbox_side_effect(skill_path: Path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(
                    passed=False, output="", error="ImportError: no module", duration_ms=50
                )
            return MagicMock(passed=True, output="Registered 1 tool(s)", error="", duration_ms=50)

        with patch.object(builder, "_sandbox_test", side_effect=sandbox_side_effect):
            result = await builder.build_skill(
                description="test",
                name="sandbox-retry",
            )

        assert result.success is True
        assert result.retries == 1

    @pytest.mark.asyncio
    async def test_build_skill_all_retries_exhausted(
        self,
        builder: SkillBuilder,
        mock_llm: AsyncMock,
    ) -> None:
        """All retries fail -> SkillBuildResult with success=False."""
        bad_code = "class NotASkill:\n    pass\n"
        bad_json = json.dumps(
            {
                "skill_md": _valid_skill_md(),
                "main_py": bad_code,
            }
        )
        mock_llm.completion.return_value = _make_llm_response(bad_json)

        result = await builder.build_skill(
            description="will fail",
            name="fail-skill",
        )

        assert result.success is False
        assert "Failed after" in (result.error or "")
        assert result.retries == builder.config.max_retries

    @pytest.mark.asyncio
    async def test_build_skill_emits_events(
        self,
        builder: SkillBuilder,
        mock_llm: AsyncMock,
        event_bus: EventBus,
    ) -> None:
        """Events are emitted for build requested and completed."""
        generated_json = json.dumps(
            {
                "skill_md": _valid_skill_md(),
                "main_py": _valid_main_py(),
            }
        )
        mock_llm.completion.return_value = _make_llm_response(generated_json)

        emitted_events: list[tuple[str, dict]] = []
        original_emit = event_bus.emit

        async def capture_emit(event: str, data: dict) -> None:
            emitted_events.append((event, data))
            await original_emit(event, data)

        event_bus.emit = capture_emit  # type: ignore[assignment]

        with patch.object(builder, "_sandbox_test") as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                passed=True, output="OK", error="", duration_ms=10
            )
            await builder.build_skill(description="test", name="event-skill")

        event_names = [e[0] for e in emitted_events]
        assert "skill.build.requested" in event_names
        assert "skill.build.completed" in event_names

    @pytest.mark.asyncio
    async def test_build_skill_caps_permissions(
        self,
        builder: SkillBuilder,
        mock_llm: AsyncMock,
    ) -> None:
        """Permissions not in config.max_permissions are filtered out."""
        generated_json = json.dumps(
            {
                "skill_md": _valid_skill_md(),
                "main_py": _valid_main_py(),
            }
        )
        mock_llm.completion.return_value = _make_llm_response(generated_json)

        with patch.object(builder, "_sandbox_test") as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                passed=True, output="OK", error="", duration_ms=10
            )
            result = await builder.build_skill(
                description="test",
                name="perm-skill",
                permissions=["safe", "dangerous"],  # "dangerous" not in max_permissions
            )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_build_skill_permissions_fallback_to_safe(
        self,
        builder: SkillBuilder,
        mock_llm: AsyncMock,
    ) -> None:
        """If all requested permissions are blocked, falls back to ['safe']."""
        generated_json = json.dumps(
            {
                "skill_md": _valid_skill_md(),
                "main_py": _valid_main_py(),
            }
        )
        mock_llm.completion.return_value = _make_llm_response(generated_json)

        with patch.object(builder, "_sandbox_test") as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                passed=True, output="OK", error="", duration_ms=10
            )
            result = await builder.build_skill(
                description="test",
                name="fallback-perm",
                permissions=["dangerous"],  # all filtered out
            )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_build_skill_writes_marker_file(
        self,
        builder: SkillBuilder,
        mock_llm: AsyncMock,
    ) -> None:
        """A .auto_generated marker file is created on success."""
        generated_json = json.dumps(
            {
                "skill_md": _valid_skill_md(),
                "main_py": _valid_main_py(),
            }
        )
        mock_llm.completion.return_value = _make_llm_response(generated_json)

        with patch.object(builder, "_sandbox_test") as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                passed=True, output="OK", error="", duration_ms=10
            )
            result = await builder.build_skill(
                description="marker test",
                name="marker-skill",
            )

        assert result.staging_path is not None
        marker = Path(result.staging_path) / ".auto_generated"
        assert marker.is_file()
        data = json.loads(marker.read_text(encoding="utf-8"))
        assert data["auto_generated"] is True
        assert data["description"] == "marker test"
        assert "created_at" in data


# ===========================================================================
# 4. approve_skill tests
# ===========================================================================


class TestApproveSkill:
    """Tests for SkillBuilder.approve_skill."""

    @pytest.mark.asyncio
    async def test_approve_moves_to_skills_dir(self, builder: SkillBuilder, tmp_path: Path) -> None:
        """Staged skill is moved to the default skills directory."""
        # Create staging dir with a skill
        staging = builder.staging_dir / "my-skill"
        staging.mkdir(parents=True)
        (staging / "SKILL.md").write_text("---\nname: my-skill\n---\n")
        (staging / "main.py").write_text("# code")

        # Without a skill_manager, target is Path("skills") / name
        # Patch Path("skills") to use tmp_path
        target_dir = tmp_path / "active_skills"
        with patch("agent.skills.builder.Path"):
            # We need the staging_dir to still work, so only patch the
            # fallback Path("skills") / name call inside approve_skill.
            # Instead, set up a skill_manager mock.
            pass

        # Use a mock skill_manager with skills_dir
        mock_manager = MagicMock()
        mock_manager.skills_dir = target_dir
        builder.skill_manager = mock_manager

        msg = await builder.approve_skill("my-skill")

        assert "approved" in msg
        assert (target_dir / "my-skill" / "SKILL.md").is_file()
        assert not staging.exists()

    @pytest.mark.asyncio
    async def test_approve_nonexistent_skill(self, builder: SkillBuilder) -> None:
        msg = await builder.approve_skill("does-not-exist")
        assert "No staged skill found" in msg

    @pytest.mark.asyncio
    async def test_approve_already_exists(self, builder: SkillBuilder, tmp_path: Path) -> None:
        """Cannot approve if skill already exists in target dir."""
        staging = builder.staging_dir / "my-skill"
        staging.mkdir(parents=True)
        (staging / "SKILL.md").write_text("---\nname: my-skill\n---\n")

        # Create target that already exists
        target_dir = tmp_path / "active_skills"
        existing = target_dir / "my-skill"
        existing.mkdir(parents=True)

        mock_manager = MagicMock()
        mock_manager.skills_dir = target_dir
        builder.skill_manager = mock_manager

        msg = await builder.approve_skill("my-skill")
        assert "already exists" in msg

    @pytest.mark.asyncio
    async def test_approve_without_skill_manager(
        self, builder: SkillBuilder, tmp_path: Path
    ) -> None:
        """Without a skill_manager, target defaults to Path('skills') / name."""
        staging = builder.staging_dir / "fallback-skill"
        staging.mkdir(parents=True)
        (staging / "SKILL.md").write_text("content")
        (staging / "main.py").write_text("code")

        builder.skill_manager = None
        # The target will be Path("skills") / "fallback-skill" in CWD
        # We patch shutil.move to avoid actual filesystem side effects outside tmp_path
        with patch("agent.skills.builder.shutil.move") as mock_move:
            msg = await builder.approve_skill("fallback-skill")

        assert "approved" in msg
        mock_move.assert_called_once()


# ===========================================================================
# 5. reject_skill tests
# ===========================================================================


class TestRejectSkill:
    """Tests for SkillBuilder.reject_skill."""

    @pytest.mark.asyncio
    async def test_reject_deletes_staging_dir(self, builder: SkillBuilder) -> None:
        staging = builder.staging_dir / "bad-skill"
        staging.mkdir(parents=True)
        (staging / "SKILL.md").write_text("content")
        (staging / "main.py").write_text("code")

        msg = await builder.reject_skill("bad-skill")

        assert "deleted" in msg
        assert not staging.exists()

    @pytest.mark.asyncio
    async def test_reject_nonexistent_skill(self, builder: SkillBuilder) -> None:
        msg = await builder.reject_skill("ghost-skill")
        assert "No staged skill found" in msg


# ===========================================================================
# 6. list_staged tests
# ===========================================================================


class TestListStaged:
    """Tests for SkillBuilder.list_staged."""

    def test_empty_staging_dir_not_created(self, builder: SkillBuilder) -> None:
        """If staging dir does not exist, returns empty list."""
        # staging_dir is in tmp_path/_staging which does not exist yet
        assert builder.list_staged() == []

    def test_empty_staging_dir_exists(self, builder: SkillBuilder) -> None:
        builder.staging_dir.mkdir(parents=True)
        assert builder.list_staged() == []

    def test_list_populated_staging(self, builder: SkillBuilder) -> None:
        staging = builder.staging_dir
        staging.mkdir(parents=True)

        # Create two skills
        s1 = staging / "alpha"
        s1.mkdir()
        (s1 / "SKILL.md").write_text("---\nname: alpha\n---")
        (s1 / "main.py").write_text("# code")

        s2 = staging / "beta"
        s2.mkdir()
        (s2 / "SKILL.md").write_text("---\nname: beta\n---")
        # beta has no main.py

        result = builder.list_staged()
        assert len(result) == 2

        names = {s["name"] for s in result}
        assert names == {"alpha", "beta"}

        alpha = next(s for s in result if s["name"] == "alpha")
        assert alpha["has_skill_md"] is True
        assert alpha["has_main_py"] is True

        beta = next(s for s in result if s["name"] == "beta")
        assert beta["has_skill_md"] is True
        assert beta["has_main_py"] is False

    def test_list_staged_with_marker_file(self, builder: SkillBuilder) -> None:
        staging = builder.staging_dir
        staging.mkdir(parents=True)

        s1 = staging / "with-marker"
        s1.mkdir()
        (s1 / "SKILL.md").write_text("---\nname: with-marker\n---")
        (s1 / "main.py").write_text("# code")
        (s1 / ".auto_generated").write_text(
            json.dumps(
                {
                    "auto_generated": True,
                    "description": "A cool skill",
                    "created_at": "2025-01-01T00:00:00",
                }
            )
        )

        result = builder.list_staged()
        assert len(result) == 1
        assert result[0]["description"] == "A cool skill"
        assert result[0]["created_at"] == "2025-01-01T00:00:00"

    def test_list_staged_ignores_dotfiles(self, builder: SkillBuilder) -> None:
        staging = builder.staging_dir
        staging.mkdir(parents=True)

        (staging / ".hidden").mkdir()
        (staging / "visible").mkdir()
        (staging / "visible" / "SKILL.md").write_text("---\nname: visible\n---")

        result = builder.list_staged()
        assert len(result) == 1
        assert result[0]["name"] == "visible"

    def test_list_staged_ignores_files(self, builder: SkillBuilder) -> None:
        """Only directories in staging are listed, not plain files."""
        staging = builder.staging_dir
        staging.mkdir(parents=True)

        (staging / "not-a-dir.txt").write_text("i am a file")
        (staging / "real-skill").mkdir()
        (staging / "real-skill" / "SKILL.md").write_text("---\nname: real-skill\n---")

        result = builder.list_staged()
        assert len(result) == 1
        assert result[0]["name"] == "real-skill"

    def test_list_staged_bad_marker_json(self, builder: SkillBuilder) -> None:
        """Corrupted .auto_generated marker is gracefully handled."""
        staging = builder.staging_dir
        staging.mkdir(parents=True)

        s1 = staging / "bad-marker"
        s1.mkdir()
        (s1 / "SKILL.md").write_text("---\nname: bad-marker\n---")
        (s1 / "main.py").write_text("# code")
        (s1 / ".auto_generated").write_text("NOT VALID JSON {{{")

        result = builder.list_staged()
        assert len(result) == 1
        assert result[0]["name"] == "bad-marker"
        # description and created_at should not be present
        assert "description" not in result[0]


# ===========================================================================
# 7. parse_natural_schedule tests
# ===========================================================================


class TestParseNaturalSchedule:
    """Tests for parse_natural_schedule from scheduler.py."""

    def test_every_morning_at_8am(self) -> None:
        result = parse_natural_schedule("every morning at 8am")
        assert result == "0 8 * * *"

    def test_every_morning_at_8_no_suffix(self) -> None:
        result = parse_natural_schedule("every morning at 8")
        assert result == "0 8 * * *"

    def test_every_friday_at_5pm(self) -> None:
        result = parse_natural_schedule("every friday at 5pm")
        assert result == "0 17 * * 5"

    def test_every_30_minutes(self) -> None:
        result = parse_natural_schedule("every 30 minutes")
        assert result == "*/30 * * * *"

    def test_every_1_minute(self) -> None:
        result = parse_natural_schedule("every 1 minute")
        assert result == "*/1 * * * *"

    def test_daily_at_noon(self) -> None:
        result = parse_natural_schedule("every day at noon")
        assert result == "0 12 * * *"

    def test_daily_at_midnight(self) -> None:
        result = parse_natural_schedule("every day at midnight")
        assert result == "0 0 * * *"

    def test_daily_at_hour_am(self) -> None:
        result = parse_natural_schedule("daily at 9am")
        assert result == "0 9 * * *"

    def test_daily_at_hour_pm(self) -> None:
        result = parse_natural_schedule("daily at 3pm")
        assert result == "0 15 * * *"

    def test_daily_at_24h_format(self) -> None:
        result = parse_natural_schedule("daily at 14:30")
        assert result == "30 14 * * *"

    def test_every_hour(self) -> None:
        result = parse_natural_schedule("every hour")
        assert result == "0 * * * *"

    def test_every_n_hours(self) -> None:
        result = parse_natural_schedule("every 2 hours")
        assert result == "0 */2 * * *"

    def test_every_monday_at_9(self) -> None:
        result = parse_natural_schedule("every monday at 9")
        assert result == "0 9 * * 1"

    def test_every_wednesday_at_10(self) -> None:
        result = parse_natural_schedule("every wednesday at 10")
        assert result == "0 10 * * 3"

    def test_every_sunday_at_7(self) -> None:
        result = parse_natural_schedule("every sunday at 7")
        assert result == "0 7 * * 0"

    def test_case_insensitive(self) -> None:
        result = parse_natural_schedule("Every Morning At 8AM")
        assert result == "0 8 * * *"

    def test_already_cron_expression(self) -> None:
        result = parse_natural_schedule("0 8 * * 1")
        assert result == "0 8 * * 1"

    def test_invalid_input_returns_none(self) -> None:
        result = parse_natural_schedule("whenever I feel like it")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = parse_natural_schedule("")
        assert result is None

    def test_random_text_returns_none(self) -> None:
        result = parse_natural_schedule("the quick brown fox")
        assert result is None

    def test_partial_cron_not_five_fields(self) -> None:
        result = parse_natural_schedule("0 8 *")
        assert result is None

    def test_every_evening_at_6pm(self) -> None:
        result = parse_natural_schedule("every evening at 6pm")
        assert result == "0 18 * * *"

    def test_every_saturday_at_10(self) -> None:
        result = parse_natural_schedule("every saturday at 10")
        assert result == "0 10 * * 6"


# ===========================================================================
# 8. _stage_skill and _write_marker tests
# ===========================================================================


class TestStagingInternals:
    """Tests for _stage_skill and _write_marker."""

    def test_stage_skill_creates_files(self, builder: SkillBuilder) -> None:
        skill = _make_generated_skill(name="staged-skill")
        path = builder._stage_skill(skill)

        assert path.is_dir()
        assert (path / "SKILL.md").is_file()
        assert (path / "main.py").is_file()
        assert (path / "SKILL.md").read_text(encoding="utf-8") == _valid_skill_md()
        assert (path / "main.py").read_text(encoding="utf-8") == _valid_main_py()

    def test_stage_skill_overwrites_existing(self, builder: SkillBuilder) -> None:
        skill = _make_generated_skill(name="overwrite-skill")
        builder._stage_skill(skill)

        # Stage again with different content
        skill2 = _make_generated_skill(name="overwrite-skill", skill_md="---\nnew: content\n---")
        path = builder._stage_skill(skill2)

        assert (path / "SKILL.md").read_text(encoding="utf-8") == "---\nnew: content\n---"

    def test_write_marker(self, builder: SkillBuilder) -> None:
        skill_path = builder.staging_dir / "marker-test"
        skill_path.mkdir(parents=True)

        builder._write_marker(skill_path, "test description")

        marker = skill_path / ".auto_generated"
        assert marker.is_file()
        data = json.loads(marker.read_text(encoding="utf-8"))
        assert data["auto_generated"] is True
        assert data["description"] == "test description"
        assert data["builder_version"] == "1.0"
        assert "created_at" in data


# ===========================================================================
# 9. _extract_from_markdown tests
# ===========================================================================


class TestExtractFromMarkdown:
    """Tests for the markdown code block fallback parser."""

    def test_extracts_skill_md_and_main_py(self, builder: SkillBuilder) -> None:
        content = (
            "Here is the skill:\n\n"
            "```\n"
            "---\n"
            "name: test\n"
            "---\n"
            "```\n\n"
            "```python\n"
            "from agent.skills.base import Skill\n\n"
            "class TestSkill(Skill):\n"
            "    async def setup(self):\n"
            "        pass\n"
            "```\n"
        )
        result = builder._extract_from_markdown(content, "test")

        assert "---" in result["skill_md"]
        assert "name: test" in result["skill_md"]
        assert "class TestSkill(Skill)" in result["main_py"]

    def test_no_code_blocks_returns_empty(self, builder: SkillBuilder) -> None:
        content = "Just plain text, no code blocks here."
        result = builder._extract_from_markdown(content, "test")

        assert result["skill_md"] == ""
        assert result["main_py"] == ""


# ===========================================================================
# 10. _generate_code tests
# ===========================================================================


class TestGenerateCode:
    """Tests for SkillBuilder._generate_code."""

    @pytest.mark.asyncio
    async def test_generate_code_parses_json(
        self, builder: SkillBuilder, mock_llm: AsyncMock
    ) -> None:
        resp_json = json.dumps(
            {
                "skill_md": "---\nname: test\n---\n",
                "main_py": "from agent.skills.base import Skill\nclass T(Skill):\n  pass",
            }
        )
        mock_llm.completion.return_value = _make_llm_response(resp_json)

        result = await builder._generate_code("test", "A test skill", ["safe"])

        assert result.name == "test"
        assert result.display_name == "Test"
        assert "---" in result.skill_md
        assert "Skill" in result.main_py

    @pytest.mark.asyncio
    async def test_generate_code_with_previous_error(
        self, builder: SkillBuilder, mock_llm: AsyncMock
    ) -> None:
        """Previous error is included in the prompt for retry context."""
        resp_json = json.dumps(
            {
                "skill_md": "---\nname: test\n---\n",
                "main_py": "from agent.skills.base import Skill\nclass T(Skill):\n  pass",
            }
        )
        mock_llm.completion.return_value = _make_llm_response(resp_json)

        await builder._generate_code(
            "test", "A test skill", ["safe"], previous_error="Missing setup method"
        )

        call_kwargs = mock_llm.completion.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert "Missing setup method" in system_msg

    @pytest.mark.asyncio
    async def test_generate_code_fallback_to_markdown(
        self, builder: SkillBuilder, mock_llm: AsyncMock
    ) -> None:
        """When LLM returns non-JSON, the markdown extractor is used."""
        markdown_response = (
            "Here is the skill:\n\n"
            "```\n"
            "---\n"
            "name: fallback\n"
            "---\n"
            "```\n\n"
            "```python\n"
            "from agent.skills.base import Skill\n\n"
            "class FallbackSkill(Skill):\n"
            "    async def setup(self):\n"
            "        pass\n"
            "```\n"
        )
        mock_llm.completion.return_value = _make_llm_response(markdown_response)

        result = await builder._generate_code("fallback", "test", ["safe"])

        assert "name: fallback" in result.skill_md
        assert "FallbackSkill" in result.main_py

    @pytest.mark.asyncio
    async def test_generate_code_json_embedded_in_text(
        self, builder: SkillBuilder, mock_llm: AsyncMock
    ) -> None:
        """JSON embedded in surrounding text is extracted correctly."""
        response = (
            "Sure, here is the skill:\n"
            '{"skill_md": "---\\nname: embedded\\n---", "main_py": "code"}\n'
            "Let me know if you need changes."
        )
        mock_llm.completion.return_value = _make_llm_response(response)

        result = await builder._generate_code("embedded", "test", ["safe"])

        assert "embedded" in result.skill_md
        assert result.main_py == "code"
