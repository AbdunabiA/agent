"""Daily Digest skill — summarize agent activity."""

from __future__ import annotations

from datetime import datetime

from agent.skills.base import Skill


class DailyDigestSkill(Skill):
    """Generate daily activity digests."""

    async def setup(self) -> None:
        """Register the digest generation tool."""
        self.register_tool(
            name="generate",
            description=(
                "Generate a daily digest summarizing agent activity. "
                "Optionally specify a date (ISO format, defaults to today)."
            ),
            function=self._generate,
            tier="safe",
        )

    def get_system_prompt_extension(self) -> str | None:
        """Inform the LLM about digest capabilities."""
        return (
            "**Daily Digest**: You can generate a daily activity summary "
            "using the daily-digest.generate tool."
        )

    async def _generate(self, date: str | None = None) -> str:
        """Generate a daily digest.

        This provides a summary template. With a database connection,
        it would query the audit log for real activity data.
        """
        target_date = date or datetime.now().strftime("%Y-%m-%d")

        lines = [
            f"# Daily Digest — {target_date}",
            "",
            "## Activity Summary",
            f"- Date: {target_date}",
            f"- Generated at: {datetime.now().isoformat()}",
            "",
            "## Sessions",
            "- No session data available (connect to database for real data)",
            "",
            "## Tools Used",
            "- No tool usage data available",
            "",
            "## Notes",
            "- This is a template digest. Connect the audit log database "
            "for detailed activity tracking.",
        ]

        return "\n".join(lines)
