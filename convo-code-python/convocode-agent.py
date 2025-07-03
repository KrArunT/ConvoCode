#!/usr/bin/env python3
"""
Multi-server MCP ↔ Ollama chat client
====================================
Launches **all** configured Model-Context-Protocol (MCP) servers, merges their
exposed tools, and lets a local **Qwen 3-0.6 b** (served by Ollama) decide which
ones to call via the OpenAI function-calling protocol.

Built-in configuration matches the JSON you provided, but you can still supply
a `servers.json` and/or choose specific blocks on the command line.

Rich-enhancements 👇
-------------------
* Colorful, readable output via `rich.print` & `Console.log`
* Live spinner while servers start with `Console.status`
* Styled user/assistant prompts with `Console.input`
* Pretty-printed warnings/errors via `Console.print_exception`

Examples
--------
```bash
# Pull model once
ollama pull qwen3:0.6b

# 1) Start *every* default server
python mcp_ollama_client.py

# 2) Load external servers.json then launch only memory + git
python mcp_ollama_client.py servers.json memory git
```
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import openai  # OpenAI-compatible client (works with Ollama)

# ──────────────────────────────────────────────────────────────────────────────
# Rich terminal helpers
# ──────────────────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from contextlib import suppress
# with suppress(RuntimeError):
#     await self.exit_stack.aclose()

console = Console()

def _banner() -> None:
    console.rule("[bold cyan]Multi-server MCP ↔ Ollama Chat Client[/]")
    console.print(
        Panel(
            "ConvoCode-Agent",
            style="bold white on dark_green",
        )
    )

# ──────────────────────────────────────────────────────────────────────────────
# Load env
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
OLLAMA_MODEL = os.getenv("MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    console.print(
        Panel(
            "Missing [bold yellow]GEMINI_API_KEY[/]. Export it or add it to your .env file",
            title="[red]Environment Error",
            style="red",
        )
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _tool_to_function(tool: Any) -> Dict[str, Any]:
    """Convert an MCP tool description into an OpenAI function spec."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            # MCP gives JSON-schema (string or dict) in inputSchema
            "parameters": (
                json.loads(tool.inputSchema)
                if isinstance(tool.inputSchema, str)
                else tool.inputSchema
            ),
        },
    }


def _expand_tilde(arg: str) -> str:
    """Expand ~ to the user home directory within command-line args."""
    return os.path.expanduser(arg) if arg.startswith("~") else arg


def _default_cfg() -> Dict[str, Any]:
    """Built-in server matrix (mirrors the latest JSON provided by the user)."""
    return {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "~/workspace/MCP/MCP_client/quickstart-resources/mcp-client-python",
                ],
            },
            "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
            "actor-critic-thinking": {
                "command": "npx",
                "args": ["-y", "mcp-server-actor-critic-thinking"],
            },
            "mcp-ssh": {"command": "npx", "args": ["@aiondadotcom/mcp-ssh"]},
            "mcp-installer": {
                "command": "npx",
                "args": ["@anaisbetts/mcp-installer"],
            },
            "memory": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-memory"],
            },
            "git": {
                "command": "uvx",
                "args": ["mcp-server-git", "--repository", "path/to/git/repo"],
            },
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR_TOKEN>"},
            },
            "mcp-server-deep-research": {
                "command": "uv",
                "args": [
                    "--directory",
                    "~/workspace/MCP/MCP_client/quickstart-resources/mcp-client-python",
                    "run",
                    "mcp-server-deep-research",
                ],
            },
        }
    }


def _load_server_config(json_path: Optional[Path]) -> Dict[str, Any]:
    """Load servers JSON or fall back to `_default_cfg`."""
    if json_path is None:
        return _default_cfg()
    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:  # pragma: no cover – best-effort fallback
        console.print(
            Panel(
                f"Could not read {json_path!s}: {exc}. Falling back to defaults.",
                title="[yellow]Warning",
                style="yellow",
            )
        )
        return _default_cfg()

# ---------------------------------------------------------------------------
# Multi-server client class
# ---------------------------------------------------------------------------

class MultiMCPClient:
    """Start multiple MCP servers and route tool calls appropriately."""

    def __init__(self) -> None:
        self.openai = openai.AsyncClient(base_url=OLLAMA_BASE_URL, api_key=API_KEY)
        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self.tool_index: Dict[str, ClientSession] = {}  # tool-name → session

    # --------------------------------------------------------------
    # Init / launch helpers
    # --------------------------------------------------------------

    async def _start_server(self, name: str, block: Dict[str, Any]) -> None:
        """Spawn one MCP server via stdio, then index its tools."""
        cmd = block["command"]
        args = [_expand_tilde(a) for a in block.get("args", [])]
        params = StdioServerParameters(command=cmd, args=args, env=block.get("env"))

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        # Map each discovered tool to this session
        for tool in (await session.list_tools()).tools:
            self.tool_index[tool.name] = session
        self.sessions.append(session)

        tool_names = ", ".join(t.name for t in (await session.list_tools()).tools)
        console.log(f"[green]✔ Started [bold]{name}[/] ({cmd}) with tools: [cyan]{tool_names}[/]")

    async def launch(self, blocks: Dict[str, Dict[str, Any]]) -> None:
        """Launch *all* server blocks concurrently."""
        await asyncio.gather(*[self._start_server(k, v) for k, v in blocks.items()])
        if not self.sessions:
            raise RuntimeError("No MCP servers started – aborting.")

    # --------------------------------------------------------------
    # LLM helper
    # --------------------------------------------------------------

    async def _llm(self, messages: List[Dict[str, Any]], tools):
        """Single call to Ollama in OpenAI format."""
        return await self.openai.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=tools or None,
            tool_choice="auto",
            max_tokens=1024,
        )

    # --------------------------------------------------------------
    # Chat processing
    # --------------------------------------------------------------

    async def process_query(self, query: str) -> str:
        """Run one user query through the LLM + tool loop."""
        messages: List[Dict[str, Any]] = [{"role": "user", "content": query}]
        oa_tools = [
            _tool_to_function(tool)
            for sess in self.sessions
            for tool in (await sess.list_tools()).tools
        ]

        response = await self._llm(messages, oa_tools)
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        outputs: List[str] = []
        while getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                session = self.tool_index.get(call.function.name)
                if session is None:
                    outputs.append(f"[Tool {call.function.name} not found]")
                    continue
                args = json.loads(call.function.arguments or "{}")
                result = await session.call_tool(call.function.name, args)
                outputs.append(result.content)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": result.content,
                    }
                )
            response = await self._llm(messages, oa_tools)
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            if msg.content:
                outputs.append(msg.content)
        return "\n".join(outputs) or msg.content or "(no response)"

    # --------------------------------------------------------------
    # REPL
    # --------------------------------------------------------------

    async def chat_loop(self) -> None:
        console.print("\n[bold green]▶ Chat started[/] (type 'exit' or 'quit' to leave)\n")
        while True:
            try:
                user_cmd = console.input("[bold cyan]You › [/]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold yellow][session terminated][/]\n")
                return
            if not user_cmd:
                continue
            if user_cmd.lower() in {"exit", "quit"}:
                console.print("[magenta][goodbye][/]\n")
                return
            try:
                console.print()  # spacer
                reply = await self.process_query(user_cmd)
                console.print("[magenta]Assistant ›[/] " + reply + "\n")
            except Exception:
                console.print_exception(show_locals=False)

    # --------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------

    async def cleanup(self) -> None:
        """Close all AsyncExitStack resources, suppressing cancel-scope errors."""
        with suppress(RuntimeError):
            await self.exit_stack.aclose()
        try:
            await self.exit_stack.aclose()
        except RuntimeError as exc:
            if "cancel scope" in str(exc):
                console.log("[yellow]Suppressed AnyIO cancel-scope error during shutdown")
            else:
                raise

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

USAGE = (
    "Usage: python mcp_ollama_client.py [servers.json] [server-name ...]\n"
    "If no server names are given, *all* in the config are started."
)

async def _main() -> None:
    _banner()

    args = sys.argv[1:]
    cfg_path: Optional[Path] = None

    # If first arg is a file, treat it as servers.json
    if args and Path(args[0]).is_file():
        cfg_path = Path(args.pop(0))
    cfg = _load_server_config(cfg_path)

    server_names: List[str] = args
    if server_names:
        unknown = [n for n in server_names if n not in cfg["mcpServers"]]
        if unknown:
            console.print(Panel(f"Unknown server IDs: {', '.join(unknown)}", style="red"))
            sys.exit(1)
        blocks = {n: cfg["mcpServers"][n] for n in server_names}
    else:
        blocks = cfg["mcpServers"]

    client = MultiMCPClient()
    try:
        with console.status("[bold green]Launching servers…[/]"):
            await client.launch(blocks)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        console.print(Panel(USAGE, title="Help", style="blue"))
        sys.exit(0)
    asyncio.run(_main())
