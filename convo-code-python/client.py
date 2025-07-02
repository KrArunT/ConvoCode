#!/usr/bin/env python3
"""
MCP‑Ollama client **(bug‑fix release)**
=====================================
* Fixes the attribute error you just hit: `ChatCompletionMessage` no longer has
  `.to_dict_recursive()` in the OpenAI 1.x SDK. We now use
  `message.model_dump(exclude_none=True)` everywhere.
* No other behaviour changes.

Run example (unchanged)
-----------------------
```
ollama pull qwen3:0.6b
python mcp_ollama_client.py servers.json filesystem
```
"""
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

import openai  # OpenAI‑compatible client (works with Ollama)

load_dotenv()

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
            "parameters": (
                json.loads(tool.inputSchema)
                if isinstance(tool.inputSchema, str)
                else tool.inputSchema
            ),
        },
    }


def _load_server_config(json_path: Optional[Path] = None) -> Dict[str, Any]:
    """Return a server‑config dict. If *json_path* is None, embed default."""
    default_cfg = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/Users/username/Desktop",
                    "/path/to/other/allowed/dir",
                ],
            }
        }
    }
    if json_path is None:
        return default_cfg
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"[warn] Could not read {json_path!s}: {exc}. Falling back to defaults.")
        return default_cfg


# ---------------------------------------------------------------------------
# Main client class
# ---------------------------------------------------------------------------

class MCPClient:
    def __init__(self):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
        self.openai = openai.AsyncClient(base_url=base_url, api_key="ollama")

        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    # ---------------------------------------------------------------------
    # Server connection logic
    # ---------------------------------------------------------------------

    async def connect(self, server_block: Dict[str, Any]):
        params = StdioServerParameters(
            command=server_block["command"],
            args=server_block.get("args", []),
            env=server_block.get("env"),
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()
        tools = [t.name for t in (await self.session.list_tools()).tools]
        print("Connected! Tools:", ", ".join(tools))

    # ---------------------------------------------------------------------
    # LLM helper
    # ---------------------------------------------------------------------

    async def _llm_call(self, messages: List[Dict[str, Any]], tools):
        return await self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            tool_choice="auto",
            max_tokens=1024,
        )

    # ---------------------------------------------------------------------
    # Main user‑query handler
    # ---------------------------------------------------------------------

    async def process_query(self, query: str) -> str:
        if self.session is None:
            raise RuntimeError("Server not connected.")

        messages = [{"role": "user", "content": query}]
        oa_tools = [_tool_to_function(t) for t in (await self.session.list_tools()).tools]

        response = await self._llm_call(messages, oa_tools)
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        output_chunks = []
        while getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = await self.session.call_tool(call.function.name, args)
                output_chunks.append(result.content)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": result.content,
                    }
                )
            response = await self._llm_call(messages, oa_tools)
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            if msg.content:
                output_chunks.append(msg.content)
        return "\n".join(output_chunks) or msg.content or "(no response)"

    # ---------------------------------------------------------------------
    # REPL
    # ---------------------------------------------------------------------

    async def chat_loop(self):
        print("\n▶  Chat started (type 'quit' to exit)\n")
        while True:
            try:
                user_input = input("You › ").strip()
                if user_input.lower() in {"quit", "exit"}:
                    break
                print()
                print("Assistant ›", await self.process_query(user_input), "\n")
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as err:
                print("[error]", err)

    async def cleanup(self):
        await self.exit_stack.aclose()


# ---------------------------------------------------------------------------
# Script entry
# ---------------------------------------------------------------------------

aSYNC_DEF = "Usage: python mcp_ollama_client.py [servers.json] <server‑name>"

async def _main():
    args = sys.argv[1:]
    if not args:
        print(aSYNC_DEF)
        sys.exit(1)

    if Path(args[0]).is_file():
        cfg_path = Path(args[0])
        server_name = args[1] if len(args) > 1 else "filesystem"
        cfg = _load_server_config(cfg_path)
    else:
        cfg_path = None
        server_name = args[0]
        cfg = _load_server_config()

    try:
        block = cfg["mcpServers"][server_name]
    except KeyError:
        print(f"Server '{server_name}' not found in config.")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect(block)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(_main())
