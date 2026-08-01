import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_project_documentation_contract_is_complete(self) -> None:
        required = (
            "AGENTS.md",
            "CONTEXT.md",
            "PROJECT.md",
            "README.md",
            "SECURITY.md",
            "TESTING.md",
            "docs/agents/domain.md",
            "docs/agents/issue-tracker.md",
            "docs/agents/triage-labels.md",
        )
        for relative_path in required:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())

        project = (ROOT / "PROJECT.md").read_text(encoding="utf-8")
        testing = (ROOT / "TESTING.md").read_text(encoding="utf-8")
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        normalized_project = " ".join(project.split())

        self.assertIn("MCP-first", project)
        self.assertIn("Deployment ownership: none", project)
        self.assertIn("Pugnet", project)
        self.assertIn("isolated", project)
        self.assertIn("279 passed", normalized_project)
        self.assertIn("18 skipped", normalized_project)
        self.assertIn("224.94 seconds", normalized_project)
        self.assertIn("35.34 seconds", normalized_project)
        self.assertIn("portable", testing.lower())
        self.assertIn("300-second", testing)
        self.assertIn("## Agent skills", agents)

    def test_inherited_review_frontend_has_repository_local_identity(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
        package_lock = json.loads((ROOT / "package-lock.json").read_text(encoding="utf-8"))

        self.assertEqual(package["name"], "woodblock-mcp-review")
        self.assertEqual(
            package["repository"]["url"],
            "https://github.com/ReidSurmeier/woodblock-reidsurmeier-wtf.git",
        )
        self.assertEqual(
            package["bugs"]["url"],
            "https://github.com/ReidSurmeier/woodblock-reidsurmeier-wtf/issues",
        )
        self.assertNotIn("homepage", package)
        self.assertEqual(package_lock["name"], package["name"])
        self.assertEqual(package_lock["packages"][""]["name"], package["name"])

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Inherited review frontend", readme)
        self.assertIn("not an active deployment", readme)

    def test_validation_and_frontend_healthcheck_are_reproducible(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "v23.yml").read_text(encoding="utf-8")
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertNotIn("actions/checkout@v", workflow)
        self.assertNotIn("actions/setup-python@v", workflow)
        self.assertNotIn("actions/setup-node@v", workflow)
        self.assertIn("ruff==0.15.20", workflow)
        self.assertIn('python-version: ["3.11", "3.12"]', workflow)
        self.assertIn("npm audit --audit-level=low", workflow)
        self.assertEqual(package["scripts"]["build"], "next build --webpack")
        self.assertIn("node", compose)
        self.assertTrue((ROOT / "eslint.config.mjs").is_file())
        self.assertTrue((ROOT / "src" / "instrumentation.ts").is_file())
        self.assertTrue((ROOT / "src" / "instrumentation-client.ts").is_file())
        self.assertTrue((ROOT / "src" / "app" / "global-error.tsx").is_file())
        for stale_sentry_config in (
            "sentry.client.config.ts",
            "sentry.edge.config.ts",
            "sentry.server.config.ts",
        ):
            self.assertFalse((ROOT / stale_sentry_config).exists())
        frontend_healthcheck = compose.split("woodblock-frontend:", maxsplit=1)[1]
        self.assertNotIn("curl -fsS http://localhost:3000/", frontend_healthcheck)

    def test_current_operator_transport_is_orca_managed_pugnet(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        project = (ROOT / "PROJECT.md").read_text(encoding="utf-8")
        historical_transport = (
            ROOT / "docs" / "adr" / "0003-stdio-over-ssh-transport.md"
        ).read_text(encoding="utf-8")
        current_transport = (
            ROOT / "docs" / "adr" / "0006-orca-managed-pugnet-execution.md"
        )

        self.assertTrue(current_transport.is_file())
        self.assertIn("Status: superseded", historical_transport)
        self.assertIn("ADR-0006", historical_transport)
        self.assertIn("Orca-managed Pugnet", readme)
        self.assertIn("Orca-managed Pugnet", project)
        self.assertNotIn("claude mcp add woodblock_stack", readme)
        self.assertNotIn("100.67.23.102", readme)

    def test_solver_backed_hitl_fixture_reuses_its_plan_pair(self) -> None:
        hitl_tests = (
            ROOT / "backend" / "tests" / "v23" / "direct" / "test_d14d_hitl_real.py"
        ).read_text(encoding="utf-8")

        self.assertIn('@pytest.fixture(scope="module")', hitl_tests)
        self.assertIn("def generated_two_plans", hitl_tests)

    def test_solver_extras_use_the_available_mixbox_release_line(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertNotIn('"mixbox>=2.0"', pyproject)
        self.assertEqual(pyproject.count('"mixbox>=1.0.5,<2"'), 2)


if __name__ == "__main__":
    unittest.main()
