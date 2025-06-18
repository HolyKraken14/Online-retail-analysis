import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# ----------------- New sales_trend tool -----------------

def sales_trend(
    chat_session_id: str,
    frequency: str = "daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate sales over time with flexible frequency and filters.

    Parameters
    ----------
    chat_session_id : str
        Session identifier whose uploaded CSVs will be analysed.
    frequency : str, optional
        One of 'daily', 'weekly', 'monthly', 'quarterly'. Defaults to 'daily'.
    start_date, end_date : str, optional
        Inclusive ISO-8601 dates (YYYY-MM-DD). If omitted, use full range.
    filters : dict, optional
        Additional equality filters, e.g. {"Gender": "Female"}.

    Returns
    -------
    dict
        {"columns": [time_bucket, "TotalSales"], "data": [[bucket, total], ...]}
    """
    freq_map = {
        "daily": "D",
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
    }
    if frequency not in freq_map:
        raise HTTPException(status_code=400, detail="Invalid frequency; choose daily/weekly/monthly/quarterly")

    df = get_dataframe_from_session(chat_session_id)
    if "InvoiceDate" not in df.columns or "Quantity" not in df.columns or "UnitPrice" not in df.columns:
        raise HTTPException(status_code=400, detail="Dataset must contain InvoiceDate, Quantity and UnitPrice columns")

    # Parse date column
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Apply date range filter
    if start_date:
        df = df[df["InvoiceDate"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["InvoiceDate"] <= pd.to_datetime(end_date)]

    # Apply arbitrary column filters
    if filters:
        for col, val in filters.items():
            if col not in df.columns:
                continue
            if isinstance(val, list):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]

    # Compute sales amount
    df["Sales"] = df["Quantity"] * df["UnitPrice"]

    # Resample / group by frequency
    bucket = df["InvoiceDate"].dt.to_period(freq_map[frequency]).dt.to_timestamp()
    agg = df.groupby(bucket)["Sales"].sum().reset_index()
    agg.columns = ["Period", "TotalSales"]

    # Convert to serialisable format
    data_rows = [[str(row["Period"].date()), float(row["TotalSales"])] for _, row in agg.iterrows()]
    return {
        "columns": ["Period", "TotalSales"],
        "data": data_rows,
        "total_rows": len(data_rows),
    }

# ----------------- end new tool -----------------

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
            
            try:
                # --- Normalize date column for SQLite compatibility ---
                if 'InvoiceDate' in df.columns:
                    try:
                        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception as _e:
                        pass  # Leave as-is if conversion fails
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
     "sales_trend": {
         "func": sales_trend,
         "description": "Aggregate sales over time with optional date range and column filters.",
         "parameters": _build_parameters_schema(sales_trend),
     }
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