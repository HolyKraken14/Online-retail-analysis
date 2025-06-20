import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import polars as pl
import json
import sqlite3
import tempfile
import re  # Added for MySQL→SQLite regex translation
import os
from sqlalchemy.orm import Session
from models import Message
from database import SessionLocal
from sqlalchemy import cast, Text
from sqlalchemy import func
# Add these imports at the top

from models import UploadedFile

# --- Application Setup ---
app = FastAPI(
    title="MCP Tool Server",
    description="A dedicated server that executes predefined data analysis tools.",
)

# --- CORS Middleware & Auth Middleware ---
ALLOWED_ORIGINS = [
    "https://chat.openai.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# MCP API-key auth (set MCP_API_KEY in environment)
API_KEY = os.getenv("MCP_API_KEY")

@app.middleware("http")
async def mcp_auth(request, call_next):
    # Protect all /~mcp routes in production
    if request.url.path.startswith("/~mcp"):
        if API_KEY:  # Only enforce if key is set
            auth_header = request.headers.get("Authorization", "")
            if auth_header != f"Bearer {API_KEY}":
                return JSONResponse(status_code=403, content={"detail": "Invalid or missing API key"})
    return await call_next(request)

# --- Tool Definitions ---

# ------------------ MCP integration helpers ------------------
from typing import get_type_hints

class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


import inspect

def _build_parameters_schema(fn) -> Dict[str, Any]:
    """Create a minimal JSON-Schema for a callable's signature usable by MCP."""
    hints = get_type_hints(fn)
    # Remove return annotation if present
    hints.pop("return", None)

    props: Dict[str, Any] = {}
    required: list[str] = []
    py_to_json = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array", "items": {"type": "string"}},  # default list mapping
        dict: {"type": "object"},
    }
    sig = inspect.signature(fn)
    for name, _typ in hints.items():
        schema = py_to_json.get(_typ, {"type": "string"}).copy()
        props[name] = schema
        if sig.parameters[name].default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": props,
        "required": required,
    }

# Placeholder; will be populated after tool function definitions to avoid forward references
_available_tools: Dict[str, Dict[str, Any]] = {}

def _get_tool_callable(name: str):
    entry = _available_tools.get(name)
    return entry["func"] if entry else None

# ------------------ end MCP helpers ------------------


def summarize_data(file_locations: List[str], chat_session_id: Optional[str] = None) -> str:
    """Return a brief text summary of the dataset."""
    try:
        dfs = [pd.read_csv(p) for p in file_locations]
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        rows, cols = df.shape
        miss = df.isna().sum().to_dict()

        summary = [f"Dataset has {rows:,} rows and {cols:,} columns."]
        summary.append("Missing values (per column):")
        for c, n in list(miss.items())[:5]:
            summary.append(f" • {c}: {n} missing")

        summary.append("\nSample unique values:")
        for c in df.columns[:5]:
            vals = [str(v) for v in df[c].dropna().unique()[:3]]
            summary.append(f" • {c}: {', '.join(vals)}")

        return "\n".join(summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in summarize_data: {e}")


def get_dataframe_from_session(chat_session_id: str) -> pd.DataFrame:
    """Get combined DataFrame from all files in a chat session"""
    db = SessionLocal()
    try:
        # Get all files for this session
        files = db.query(UploadedFile).filter(
            UploadedFile.chat_session_id == chat_session_id
        ).all()
        
        if not files:
            raise ValueError(f"No files found for session {chat_session_id}")
        
        # Read and combine all CSVs
        dfs = [pd.read_csv(f.file_path) for f in files]
        return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    finally:
        db.close()



# Update the tool definitions
def execute_sql_query(chat_session_id: str, sql_query: str) -> Dict:
    """Execute SQL query on session data using SQLite."""
    try:
        print(f"Executing SQL query for session: {chat_session_id}")
        print(f"Raw SQL Query: {repr(sql_query)}")
        
        # Get DataFrame from session
        df = get_dataframe_from_session(chat_session_id)
        print(f"Loaded dataframe with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Clean up SQL query
        sql_query = sql_query.strip()
        sql_statements = [stmt.strip() for stmt in sql_query.split(';') if stmt.strip()]
        
        if len(sql_statements) == 0:
            raise ValueError("No valid SQL statement found")
        
        sql_query = sql_statements[0]
        print(f"Cleaned SQL Query (pre-translate): {sql_query}")

        # --- translate common MySQL date helpers to SQLite equivalents ---
        def _mysql_to_sqlite(q: str) -> str:
            """Best-effort string replace for MySQL date helpers so LLM SQL still runs on SQLite."""
            # STR_TO_DATE(col, '%m/%d/%Y %H:%i') → col  (column already ISO text)
            q = re.sub(r"STR_TO_DATE\s*\(\s*([A-Za-z0-9_\.]+)\s*,\s*'[^']+'\s*\)", r"\1", q, flags=re.IGNORECASE)
            # DATE_FORMAT(col, '%Y-%m') → strftime('%Y-%m', col)
            q = re.sub(r"DATE_FORMAT\s*\(\s*([A-Za-z0-9_\.]+)\s*,\s*'(%Y[^']+)'\s*\)", r"strftime('\2', \1)", q, flags=re.IGNORECASE)
            # YEARWEEK(col) or YEARWEEK(col,0) → strftime('%Y-%W', col)
            q = re.sub(r"YEARWEEK\s*\(\s*([A-Za-z0-9_\.]+)(?:\s*,\s*\d+)?\s*\)", r"strftime('%Y-%W', \1)", q, flags=re.IGNORECASE)
            return q
        sql_query = _mysql_to_sqlite(sql_query)
        print(f"SQL after MySQL→SQLite translation: {sql_query}")
        
        # Create temporary SQLite database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            conn = sqlite3.connect(tmp.name)
            # Register custom MEDIAN aggregate so that LLM-generated SQL with MEDIAN works
            # without breaking existing queries. This is lightweight and scoped to the
            # in-memory temp DB, so normal flow is unchanged.
            class _MedianAgg:
                def __init__(self):
                    self.values = []
                def step(self, value):
                    if value is None:
                        return
                    try:
                        self.values.append(float(value))
                    except (TypeError, ValueError):
                        pass
                def finalize(self):
                    if not self.values:
                        return None
                    self.values.sort()
                    n = len(self.values)
                    mid = n // 2
                    if n % 2 == 1:
                        return self.values[mid]
                    return (self.values[mid - 1] + self.values[mid]) / 2
            try:
                conn.create_aggregate("median", 1, _MedianAgg)
                conn.create_aggregate("MEDIAN", 1, _MedianAgg)  # case-insensitive helper
            except (sqlite3.OperationalError, AttributeError):
                # If aggregate already exists or cannot be registered, ignore gracefully
                pass
            
            try:
                # --- Normalize date column for SQLite compatibility ---
                if 'InvoiceDate' in df.columns:
                    try:
                        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception as _e:
                        pass  # Leave as-is if conversion fails
                # --- Safen column names (spaces / special chars) for SQLite ---
                original_columns = df.columns.tolist()
                safe_columns = []
                col_mapping: dict[str, str] = {}
                for col in original_columns:
                    safe_col = re.sub(r'\W+', '_', col).strip('_')
                    col_mapping[col] = safe_col
                    safe_columns.append(safe_col)
                if any(orig != safe for orig, safe in zip(original_columns, safe_columns)):
                    df.columns = safe_columns
                    # Replace references in SQL query so it matches renamed columns
                    for orig, safe in col_mapping.items():
                        if orig != safe:
                            sql_query = re.sub(rf"\b{re.escape(orig)}\b", safe, sql_query)
                    print(f"SQL after column-name sanitization: {sql_query}")
                df.to_sql('csv_table', conn, index=False, if_exists='replace')
                cursor = conn.cursor()
                cursor.execute(sql_query)
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                result_data = [dict(zip(columns, row)) for row in rows]
                
                print(f"Query result: {len(result_data)} rows, {len(columns)} columns")
                
                return {
                    "data": result_data,
                    "columns": columns,
                    "row_count": len(result_data)
                }
            finally:
                conn.close()
                try:
                    os.unlink(tmp.name)
                except:
                    pass
                    
    except Exception as e:
        print(f"Error in execute_sql_query: {str(e)}")
        print(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing SQL: {e}")

# ------------------ Register tools after definitions ------------------
_available_tools.update({
    "summarize_data": {
        "func": summarize_data,
        "description": "Return a brief textual summary of uploaded CSV data.",
        "parameters": _build_parameters_schema(summarize_data),
    },
    "execute_sql_query": {
         "func": execute_sql_query,
         "description": "Run a SQL query against session-scoped CSV data.",
         "parameters": _build_parameters_schema(execute_sql_query),
     },

})
# ---------------------------------------------------------------------

# Update the tool request model
class ToolExecutionRequest(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to execute.")
    arguments: Dict[str, Any] = Field(..., description="The arguments for the tool.")
    chat_session_id: Optional[str] | None = Field(None, description="Optional chat session ID if the tool requires it.")

# Update the execute tool endpoint
@app.post("/execute-tool")
async def execute_tool(request: ToolExecutionRequest):
    """Execute a tool on session data."""
    print(f"Received request to execute tool: {request.tool_name}")
    print(f"For session: {request.chat_session_id}")
    
    function_to_call = _get_tool_callable(request.tool_name)
    
    if not function_to_call:
        available_names = list(_available_tools.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Tool '{request.tool_name}' not found. Available tools: {available_names}"
        )
    
    try:
        # Add chat_session_id to tool arguments
        tool_args = dict(request.arguments)
        if request.chat_session_id is not None:
            tool_args["chat_session_id"] = request.chat_session_id
        result = function_to_call(**tool_args)
        print(f"Tool execution successful")
        return result
    except Exception as e:
        print(f"Tool execution failed: {str(e)}")
        raise


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "available_tools": list(_available_tools.keys())}


# ------------------ MCP endpoints ------------------

class MCPToolListResponse(BaseModel):
    tools: List[ToolSpec]


@app.get("/~mcp/tools", response_model=MCPToolListResponse)
async def list_tools():
    tool_specs = [
        ToolSpec(name=name, description=info["description"], parameters=info["parameters"])
        for name, info in _available_tools.items()
    ]
    return {"tools": tool_specs}


class MCPExecuteRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any]


@app.post("/~mcp/execute")
async def mcp_execute(req: MCPExecuteRequest):
    tool_func = _get_tool_callable(req.tool)
    if not tool_func:
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool}' not found")
    try:
        return tool_func(**req.arguments)
    except TypeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/~mcp/healthz")
async def mcp_health():
    return {"status": "healthy"}

# Serve .well-known directory if present
if os.path.isdir(".well-known"):
    app.mount("/.well-known", StaticFiles(directory=".well-known"), name="well-known")

# --- Main entry point to run the server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)