"""
Shared utilities for the agent.
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()


def print_response(message):
    """Pretty-print a Claude API response."""
    for block in message.content:
        if block.type == "text":
            print(f"\n🤖 Claude: {block.text}")
        elif block.type == "tool_use":
            print(f"\n🔧 Tool call: {block.name}({json.dumps(block.input, indent=2)})")


def print_tool_result(name, result):
    """Pretty-print a tool execution result."""
    print(f"\n📡 {name} returned: {json.dumps(result, indent=2)}")


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 60 + "\n")


def get_api_key() -> str:
    """Load and return the Anthropic API key from environment."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. "
            "Add it to your .env file or export it in your shell."
        )
    return key