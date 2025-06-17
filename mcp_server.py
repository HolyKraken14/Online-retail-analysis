# mcp_server.py - UPDATED VERSION
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import polars as pl
import json
import sqlite3
import tempfile
import os

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

def summarize_data(file_locations: List[str]) -> str:
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


def execute_sql_query(file_locations: List[str], sql_query: str) -> Dict:
    """Execute SQL query on CSV files using SQLite."""
    try:
        print(f"Executing SQL query on files: {file_locations}")
        print(f"Raw SQL Query: {repr(sql_query)}")
        
        # Check if files exist
        for file_path in file_locations:
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Load the CSV file(s)
        dfs = [pd.read_csv(p) for p in file_locations]
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        
        print(f"Loaded dataframe with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Clean up the SQL query
        sql_query = sql_query.strip()
        
        # Remove any trailing semicolons and split by semicolon to handle multiple statements
        sql_statements = [stmt.strip() for stmt in sql_query.split(';') if stmt.strip()]
        
        if len(sql_statements) == 0:
            raise ValueError("No valid SQL statement found")
        
        # Use only the first statement if multiple are present
        if len(sql_statements) > 1:
            print(f"Warning: Multiple SQL statements detected. Using only the first one.")
            sql_query = sql_statements[0]
        else:
            sql_query = sql_statements[0]
        
        print(f"Cleaned SQL Query: {sql_query}")
        
        # Create temporary SQLite database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            conn = sqlite3.connect(tmp.name)
            
            try:
                # Write DataFrame to SQLite with the expected table name
                df.to_sql('csv_table', conn, index=False, if_exists='replace')
                
                # Execute the query
                cursor = conn.cursor()
                cursor.execute(sql_query)
                
                # Fetch results
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                result_data = [dict(zip(columns, row)) for row in rows]
                
                print(f"Query result: {len(result_data)} rows, {len(columns)} columns")
                
                return {
                    "data": result_data,
                    "columns": columns,
                    "row_count": len(result_data)
                }
                
            finally:
                conn.close()
                # Clean up temp file
                try:
                    os.unlink(tmp.name)
                except:
                    pass
            
    except Exception as e:
        print(f"Error in execute_sql_query: {str(e)}")
        print(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing SQL: {e}")


def generate_echart_config(
    file_locations: List[str],
    chart_type: str,
    x_axis: Optional[str] = None,
    y_axes: Optional[List[str]] = None,
    filters: Optional[str] = None
) -> Dict:
    """Build an ECharts `option` dict for common charts."""
    try:
        # Check if files exist
        for file_path in file_locations:
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
                
        df = pd.read_csv(file_locations[0])
        if filters:
            df = df.query(filters)
        
        option: Dict[str, Any] = {"series": []}
        
        if chart_type in ("line","bar") and x_axis and y_axes:
            df_group = df.groupby(x_axis)[y_axes].sum().reset_index()
            option.update({
                "xAxis": {"type": "category", "data": df_group[x_axis].tolist()},
                "yAxis": {"type": "value"},
                "tooltip": {"trigger": "axis"},
                "legend": {"data": y_axes},
                "series": [
                    {"name": y, "type": chart_type, "data": df_group[y].tolist()}
                    for y in y_axes
                ]
            })
        elif chart_type == "pie" and x_axis and y_axes and len(y_axes)==1:
            data = (
                df.groupby(x_axis)[y_axes[0]]
                .sum().nlargest(10)
                .reset_index()
                .rename(columns={x_axis: "name", y_axes[0]: "value"})
            )
            option["series"] = [{
                "type": "pie",
                "data": data.to_dict("records"),
                "radius": "50%"
            }]
            option["tooltip"] = {"trigger": "item"}
            option["legend"] = {"orient": "vertical", "left": "left", "data": data["name"].tolist()}
        else:
            raise ValueError("Invalid chart_type or missing axes for chart generation.")

        return option
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in generate_echart_config: {e}")


# --- API Endpoint to Execute Tools ---
class ToolExecutionRequest(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to execute.")
    arguments: Dict[str, Any] = Field(..., description="The arguments for the tool.")

@app.post("/execute-tool")
async def execute_tool(request: ToolExecutionRequest):
    """
    Receives a tool name and arguments, executes the corresponding
    function, and returns the result.
    """
    print(f"Received request to execute tool: {request.tool_name}")
    print(f"Arguments: {request.arguments}")
    
    available_tools = {
        "summarize_data": summarize_data,
        "generate_echart_config": generate_echart_config,
        "execute_sql_query": execute_sql_query,  # ✅ ADDED THE MISSING TOOL
    }
    
    function_to_call = available_tools.get(request.tool_name)
    
    if not function_to_call:
        available_names = list(available_tools.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Tool '{request.tool_name}' not found. Available tools: {available_names}"
        )
        
    try:
        result = function_to_call(**request.arguments)
        print(f"Tool execution successful")
        return result
    except Exception as e:
        print(f"Tool execution failed: {str(e)}")
        raise


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "available_tools": ["summarize_data", "generate_echart_config", "execute_sql_query"]}


# --- Main entry point to run the server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)