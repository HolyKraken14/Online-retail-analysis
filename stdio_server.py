"""StdIO MCP server wrapper that exposes the same tools via stdin/stdout.

Run with:
    python stdio_server.py
"""

from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any, Optional

# Re-use existing implementations
from mcp_server import summarize_data, execute_sql_query

mcp = FastMCP(name="Retail Data Tools")


@mcp.tool()
def summarize_data_tool(file_locations: List[str], chat_session_id: Optional[str] = None) -> str:
    """Return a textual summary of uploaded CSV data."""
    return summarize_data(file_locations=file_locations, chat_session_id=chat_session_id)


@mcp.tool()
def execute_sql_query_tool(sql_query: str, chat_session_id: str) -> Dict[str, Any]:
    """Run a SQL query against session-scoped CSV data."""
    return execute_sql_query(chat_session_id=chat_session_id, sql_query=sql_query)


if __name__ == "__main__":
    # Use stdio transport so a local LLM agent (or your MCPOpenAIClient) can spawn the process
    mcp.run(transport="stdio")
