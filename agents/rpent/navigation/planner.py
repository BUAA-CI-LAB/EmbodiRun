"""Use RPent's Codex planner with only the navigation MCP tools enabled."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import openai_codex
import tomllib
from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.codex import CodexPlanner

from .toolkit import SYSTEM_PROMPT, NavigationToolkit


class NavigationPlanner(CodexPlanner):
    """Pin the RPent adapter contract and exclude local evaluator file access."""

    def __init__(self, output: Path, *, model: str, timeout_s: int = 3600):
        workspace = output / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        super().__init__(
            output_dir=str(output),
            repo_root=workspace,
            output_path=output / "planner.txt",
            model=model,
            timeout_s=timeout_s,
            reasoning_effort="low",
            dashboard_events=NullDashboardEventSink(),
        )
        self._thread_options.update(sandbox=openai_codex.Sandbox.read_only, ephemeral=True)
        self._turn_options["sandbox"] = openai_codex.Sandbox.read_only

    def _build_config(self, mcp_url):
        config = super()._build_config(mcp_url)
        binary = shutil.which("codex")
        if binary is None:
            raise RuntimeError("install and authenticate the Codex CLI before running the planner")
        overrides = list(config.config_overrides)
        overrides += [
            f"features.{name}=false"
            for name in (
                "shell_tool",
                "apply_patch_freeform",
                "apps",
                "plugins",
                "hooks",
                "memories",
                "multi_agent",
                "browser_use",
                "computer_use",
            )
        ]
        overrides += ['web_search="disabled"']
        allowed_tools = ("observe", "navigate", "update_plan", "finish")
        overrides.append(f"mcp_servers.rpent.enabled_tools={json.dumps(allowed_tools)}")
        for name in allowed_tools:
            overrides.append(f'mcp_servers.rpent.tools.{name}.approval_mode="approve"')
        # Disable user MCP integrations while retaining the CLI's existing login.
        # No credentials are transferred to the model machine.
        config_path = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
        if config_path.exists():
            user_config = tomllib.loads(config_path.read_text())
            for name in user_config.get("mcp_servers", {}):
                if name != "rpent":
                    overrides.append(f"mcp_servers.{json.dumps(name)}.enabled=false")
        command = [binary]
        for item in overrides:
            command += ["-c", item]
        command += ["app-server", "--listen", "stdio://"]
        return replace(config, launch_args_override=tuple(command))


def solve(toolkit: NavigationToolkit, instruction: str, output: Path, *, model: str, timeout_s: int = 3600):
    planner = NavigationPlanner(output, model=model, timeout_s=timeout_s)
    return planner.solve(
        system_prompt=SYSTEM_PROMPT, user_message=instruction, toolkit=toolkit, max_turns=toolkit.max_tool_calls
    )
