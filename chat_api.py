from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from jose import jwt
import os
import uuid
from datetime import datetime
import pandas as pd
import json
from typing import List, Optional
from pydantic import BaseModel
import numpy as np

from fastapi import Header
from models import User, ChatSession, Message, UploadedFile
from database import SessionLocal, Base, engine
import models
from openai import OpenAI
from dotenv import load_dotenv
from sqlalchemy import orm
from openai import OpenAI
from pydantic import Field, create_model
import httpx
from typing import Dict, Any, List

# Load .env
load_dotenv()
MCP_SERVER_URL = "http://localhost:8004"

# Ensure tables exist
Base.metadata.create_all(bind=engine)
router = APIRouter()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
UPLOAD_DIR = "uploads"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_openai_tool_config(model, **fields):
    """Helper to convert Pydantic models to OpenAI tool schema."""
    doc = model.__doc__ or ""
    pydantic_model = create_model(model.__name__, __doc__=doc, **fields)
    schema = pydantic_model.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": schema['title'],
            "description": schema['description'],
            "parameters": {k: v for k, v in schema.items() if k not in ['title', 'description']}
        }
    }

class UploadedFileSchema(BaseModel):
    id: int
    original_filename: str
    file_size: Optional[int] = None
    schema_info: Optional[Dict[str, Any]] = None
    class Config: 
        orm_mode = True

class ChatSessionSchema(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    uploaded_files: List[UploadedFileSchema] = []
    class Config: 
        orm_mode = True

class MessageResponseWithMetadata(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    class Config: 
        orm_mode = True

class UploadResponseSchema(BaseModel):
    chat_session: ChatSessionSchema
    initial_message: MessageResponseWithMetadata

class ChatSessionCreate(BaseModel): 
    title: str = "New Chat"

class MessageCreate(BaseModel): 
    content: str

# OpenAI tool configurations
SummarizeDataArgs = create_model(
    'summarize_data',
    __doc__="Return a brief text summary of the dataset: #rows, #cols, missing values, and sample unique values.",
)

# Tool args for sales trend aggregation
SalesTrendArgs = create_model(
    "sales_trend",
    __doc__="Aggregate sales over time with flexible frequency and filters.",
    frequency=(str, Field("daily", enum=["daily", "weekly", "monthly", "quarterly"])),
    start_date=(Optional[str], Field(None)),
    end_date=(Optional[str], Field(None)),
    filters=(Optional[Dict[str, Any]], Field(None)),
)

EchartArgs = create_model(
    'generate_echart_config',
    __doc__="Build an ECharts `option` dict from structured JSON data. Use this AFTER getting data from a SQL query.",
    data=(List[Dict[str, Any]], Field(..., description="The JSON data from a previous SQL query result.")),
    chat_session_id=(str, Field(..., description="The chat session ID to associate this chart with.")),
    chart_type=(str, Field(..., description="Any ECharts series type, e.g. line, bar, pie, scatter, heatmap, radar, treemap, funnel, gauge")),
    x_axis=(str, Field(..., description="The column name to use for the X-axis.")),
    y_axis=(List[str], Field(..., description="The column name(s) to use for the Y-axis.")),
    title=(Optional[str], Field(None, description="A descriptive title for the chart."))
)

SQLArgs = create_model(
    'execute_sql_query',
    __doc__="Execute a SQL query against the CSV data. The table name is `csv_table`. Use this for complex questions, joins, or specific data filtering.",
    sql_query=(str, Field(..., description="The SQL query to execute."))
)

openai_tools = [
    get_openai_tool_config(SummarizeDataArgs),
    get_openai_tool_config(SalesTrendArgs),
    get_openai_tool_config(SQLArgs),
    get_openai_tool_config(EchartArgs),
]

def get_current_user(authorization: str, db: Session):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No valid token")
    
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        user = db.query(User).filter(
            (User.oauth_sub == user_id) | (User.email == user_id)
        ).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
def create_valid_chart_config(chart_type: str, title: str, x_data: list, y_data: list) -> dict:
    return {
        "title": {
            "text": title,
            "left": "center",
            "top": 20
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {
                "type": "shadow"
            }
        },
        "grid": {
            "left": "5%",
            "right": "5%",
            "bottom": "10%",
            "containLabel": True
        },
        "xAxis": {
            "type": "category",
            "data": x_data,
            "axisTick": {
                "alignWithLabel": True
            },
            "axisLabel": {
                "rotate": 45,
                "overflow": "truncate"
            }
        },
        "yAxis": {
            "type": "value"
        },
        "series": [{
            "name": title,
            "type": chart_type,
            "data": y_data,
            "label": {
                "show": True,
                "position": "top"
            }
        }]
    }

def format_chat_history(messages: List[Message]) -> str:
    """Format chat history for LLM context"""
    history_parts = []
    for msg in messages:
        if msg.role == "user":
            history_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            # Only include the summary from metadata if available, not raw analysis results
            if msg.metadata_ and isinstance(msg.metadata_, (str, dict)):
                try:
                    metadata = json.loads(msg.metadata_) if isinstance(msg.metadata_, str) else msg.metadata_
                    if metadata.get("type") == "analysis_result" and metadata.get("summary"):
                        history_parts.append(f"Assistant: {metadata['summary']}")
                    else:
                        history_parts.append(f"Assistant: {msg.content}")
                except:
                    history_parts.append(f"Assistant: {msg.content}")
            else:
                history_parts.append(f"Assistant: {msg.content}")
    
    return "\n".join(history_parts[-10:])  # Keep last 10 exchanges for context

@router.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
    chat_session_id: Optional[str] = Form(None),
    authorization: str = Header(..., alias="Authorization"),
    db: Session = Depends(get_db)
):
    user = get_current_user(authorization, db)
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Analyze CSV schema
    try:
        df_sample = pd.read_csv(file_path, nrows=5)
        df_sample = df_sample.replace({np.nan: None})
        schema_info = {
            "columns": df_sample.columns.tolist(),
            "dtypes": df_sample.dtypes.astype(str).to_dict(),
            "shape": df_sample.shape,
            "sample_data": df_sample.to_dict('records')
        }
    except Exception as e:
        schema_info = {"error": str(e)}
    
    # Use existing session or create new one
    if chat_session_id:
        # Verify session belongs to user
        chat_session = db.query(ChatSession).filter(
            ChatSession.id == chat_session_id,
            ChatSession.user_id == user.id
        ).first()
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")
    else:
        # Create a new chat session for this file
        chat_session = ChatSession(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title=f"Analysis: {file.filename}"
        )
        db.add(chat_session)
        db.flush()
    
    # Save uploaded file with the session ID
    uploaded_file = UploadedFile(
        user_id=user.id,
        chat_session_id=chat_session.id,
        filename=filename,
        original_filename=file.filename,
        file_path=file_path,
        file_size=len(content),
        mime_type=file.content_type,
        schema_info=schema_info
    )
    
    db.add(uploaded_file)
    db.commit()
    db.refresh(uploaded_file)
    chat_session.has_file = True
    db.commit()
    db.refresh(chat_session)
    db.refresh(chat_session)
    
    # Generate summary using MCP
    try:
        async with httpx.AsyncClient() as http_client:
            tool_response = await http_client.post(
                f"{MCP_SERVER_URL}/~mcp/execute",
                json={
                    "tool": "summarize_data",
                    "arguments": {"file_locations": [file_path], "chat_session_id": chat_session.id}
                },
                timeout=60.0
            )
            tool_response.raise_for_status()
            summary = tool_response.json()
    except Exception as e:
        summary = f"Error generating summary: {str(e)}"
    
    # Create initial message if this is a new upload
    if not chat_session_id:
        welcome_content = f"I've analyzed {file.filename}. Here are the initial details:\n\n{summary}\n\nWhat would you like to explore further?"
        initial_message = Message(
            chat_session_id=chat_session.id,
            role="assistant",
            content=welcome_content
        )
        db.add(initial_message)
        db.commit()
        db.refresh(initial_message)
    
    return {
        "file_info": {
            "file_id": uploaded_file.id,
            "filename": file.filename,
            "chat_session_id": chat_session.id,
            "schema": schema_info
        },
        "summary": summary
    }

@router.get("/chat-sessions")
async def get_chat_sessions(
    authorization: str = Header(..., alias="Authorization"),
    db: Session = Depends(get_db)
):
    user = get_current_user(authorization, db)
    
    sessions = db.query(ChatSession).options(
        orm.joinedload(ChatSession.uploaded_files)
    ).filter(
        ChatSession.user_id == user.id,
        ChatSession.is_active == True
    ).order_by(ChatSession.updated_at.desc()).all()
    
    return [
        {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "has_file": session.has_file,  # Add this line
            "uploaded_files": [
                {
                    "id": f.id,
                    "original_filename": f.original_filename,
                    "file_size": f.file_size,
                    "schema_info": f.schema_info
                } for f in session.uploaded_files
            ] if session.uploaded_files else []
        }
        for session in sessions
    ]

@router.get("/chat-sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    authorization: str = Header(..., alias="Authorization"),
    db: Session = Depends(get_db)
):
    user = get_current_user(authorization, db)
    
    session = db.query(ChatSession).options(
        orm.joinedload(ChatSession.uploaded_files)
    ).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "uploaded_files": [
            {
                "id": f.id,
                "original_filename": f.original_filename,
                "file_size": f.file_size,
                "schema_info": f.schema_info
            } for f in session.uploaded_files
        ] if session.uploaded_files else []
    }

@router.post("/chat-sessions", response_model=ChatSessionSchema)
async def create_session(
    session_data: ChatSessionCreate, 
    authorization: str = Header(...), 
    db: Session = Depends(get_db)
):
    user = get_current_user(authorization, db)
    session = ChatSession(
        id=str(uuid.uuid4()), 
        user_id=user.id, 
        title=session_data.title
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

@router.get("/chat-sessions/{session_id}/messages")
async def get_messages(
    session_id: str,
    authorization: str = Header(..., alias="Authorization"),
    db: Session = Depends(get_db)
):
    user = get_current_user(authorization, db)

    # Verify session belongs to user
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    msgs = db.query(Message)\
        .filter(Message.chat_session_id == session_id)\
        .order_by(Message.created_at.asc())\
        .all()

    return [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at,
            "metadata": (
                json.loads(msg.metadata_)
                if isinstance(msg.metadata_, str) and msg.metadata_ else
                msg.metadata_
            )
        }
        for msg in msgs
    ]

@router.post(
    "/chat-sessions/{session_id}/messages",
    response_model=MessageResponseWithMetadata
)
async def add_message(
    session_id: str,
    message_data: MessageCreate,
    authorization: str = Header(..., alias="Authorization"),
    db: Session = Depends(get_db)
):
    user = get_current_user(authorization, db)

    # Verify session belongs to user
    session = db.query(ChatSession).options(
        orm.joinedload(ChatSession.uploaded_files)
    ).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # 1. Persist user's message
    user_message = Message(
        chat_session_id=session_id,
        role="user",
        content=message_data.content
    )
    db.add(user_message)
    session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user_message)

    # 2. If no file uploaded for this session, respond accordingly
    if not session.uploaded_files:
        assistant_content = "I can only analyze data if you upload a CSV file in this chat. Please upload a CSV file to get started with data analysis."
        assistant_message = Message(
            chat_session_id=session_id,
            role="assistant",
            content=assistant_content
        )
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)
        return {
            "id": assistant_message.id,
            "role": assistant_message.role,
            "content": assistant_message.content,
            "created_at": assistant_message.created_at,
            "metadata": None
        }

    # 3. Get chat history for context (excluding the current user message)
    messages_history = db.query(Message)\
        .filter(Message.chat_session_id == session_id, Message.id != user_message.id)\
        .order_by(Message.created_at.asc())\
        .all()
    
    chat_history = format_chat_history(messages_history)
    file_paths = [f.file_path for f in session.uploaded_files]

    # 4. Build enhanced prompt with chat history
    schema_info = session.uploaded_files[0].schema_info
    schema_description = json.dumps(schema_info, indent=2)

    prompt = f"""
You are a data analysis assistant with access to CSV data. Here's the context:

CHAT HISTORY:
{chat_history}

CSV SCHEMA AND SAMPLE DATA:
{schema_description}

CURRENT USER REQUEST: "{message_data.content}"

INSTRUCTIONS:
1. Consider the chat history to understand the context and previous analyses
2. The table name in SQL queries is `csv_table`
3. Generate only ONE SQL statement without semicolons
4. For "top N" queries, use appropriate GROUP BY and ORDER BY
5. Column names are case-sensitive
6. If this is a general greeting (hi, hello, help), respond conversationally
7. If this relates to previous analysis, reference it appropriately
8. **Use SQLite syntax only. Do not use DATE_TRUNC, :: , EXTRACT, etc.
RESPONSE FORMAT:
If this appears to be sales/business data analysis:
Sales Data: Yes
SQL Query: [Your SQL query here]

If not sales data or general conversation:
Sales Data: No
[Your conversational response here]
"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    response_content = response.choices[0].message.content.strip()
    print(f"OpenAI response: {response_content}")

    # 5. Handle OpenAI response
    msg = response.choices[0].message
    metadata_to_save: dict | None = None

    if msg.function_call:
        func_name = msg.function_call.name
        raw_args = msg.function_call.arguments or {}
        if isinstance(raw_args, str):
            try:
                func_args = json.loads(raw_args)
            except json.JSONDecodeError:
                func_args = {}
        else:
            func_args = raw_args

        # Always include session id so server-side tools can scope data
        func_args.setdefault("chat_session_id", session_id)

        # Branch: chart generation is handled entirely by the LLM, no MCP call
        if func_name == "generate_echart_config":
            chart_config = func_args  # model already produced final ECharts option
            assistant_content = "Here is the chart configuration for your request."
            metadata_to_save = {"type": "chart_config", "echartOption": chart_config}
        else:
            # Execute chosen tool via MCP HTTP endpoint
            try:
                async with httpx.AsyncClient() as http_client:
                    tool_resp = await http_client.post(
                        f"{MCP_SERVER_URL}/~mcp/execute",
                        json={"tool": func_name, "arguments": func_args},
                        timeout=60.0,
                    )
                    tool_resp.raise_for_status()
                    tool_result = tool_resp.json()
            except Exception as e:
                tool_result = {"error": str(e)}

            assistant_content = f"Executed {func_name}."
            metadata_to_save = {
                "type": "analysis_result",
                "tool": func_name,
                "arguments": func_args,
                "result": tool_result,
            }
    else:
        # Model responded with plain text – treat as normal assistant message
        assistant_content = msg.content.strip() or "I'm not sure how to help with that."
    metadata_to_save = None
    
    if response_content.startswith("Sales Data: No") or "Sales Data: No" in response_content:
        # Extract the conversational response
        if "Sales Data: No" in response_content:
            assistant_content = response_content.split("Sales Data: No")[1].strip()
            if not assistant_content:
                assistant_content = "I'm here to help you analyze your data. What would you like to know?"
        else:
            assistant_content = response_content
    else:
        # Extract and execute SQL
        try:
            if "SQL Query:" in response_content:
                sql_section = response_content.split("SQL Query:")[1].strip()
                # Clean up SQL formatting
                sql_section = sql_section.replace("```sql", "").replace("```", "").strip()
                sql_query = sql_section.rstrip(';')
            else:
                sql_query = None
        except Exception as e:
            print(f"Error extracting SQL: {e}")
            sql_query = None

        if sql_query:
            try:
                # Execute SQL via MCP tool with session ID
                async with httpx.AsyncClient() as http_client:
                    exec_resp = await http_client.post(
                        f"{MCP_SERVER_URL}/~mcp/execute",
                        json={
                            "tool": "execute_sql_query",
                            "arguments": {
                                "sql_query": sql_query,
                                "chat_session_id": session_id
                            }
                        },
                        timeout=60.0
                    )
                    exec_resp.raise_for_status()
                    sql_result = exec_resp.json()
                    print(f"SQL execution result: {sql_result}")

                # Generate natural language summary with chat context
                summary = await generate_summary_with_context(
                    client, message_data.content, sql_result, chat_history
                )

                # Generate chart configuration
                chart_config = None
                if sql_result.get("data") and len(sql_result["data"]) > 0:
                    cols = sql_result["columns"]
                    if len(cols) >= 2:
                        chart_prompt = f"""
                        Based on the chat history and current analysis, generate an ECharts configuration:
                        **USE SQLLite functions to generate the chart**
                        Previous Context: {chat_history[-500:]}  # Last 500 chars of history
                        Current Query: {message_data.content}
                        Columns: {cols}
                        Sample data: {json.dumps(sql_result['data'][:10])}
                        This is the JSON schema for an ECharts config—do not alter its structure or field names:
    ```json
    {{
      "title": {{ /* object */ }},
      "tooltip": {{ /* object */ }},
      "xAxis":  {{ /* object */ }},
      "yAxis":  {{ /* object */ }},
      "series": [
        {{ /* object */ }}
      ]
    }}
    ```
    Use that schema exactly when generating your output.

                        Create an appropriate chart (bar, line, pie) showing {cols[1]} by {cols[0]}.
                        Do not generate the legend for any of the charts
                        Make the title descriptive and relevant to the user's request.
                        """
                        try:
                            functions_list = [tool["function"] for tool in openai_tools]
                            chart_resp = client.chat.completions.create(
                                model="gpt-4-1106-preview",
                                messages=[{"role": "user", "content": chart_prompt}],
                                functions=functions_list,
                                function_call={"name": "generate_echart_config"}
                            )
                            
                            function_call = chart_resp.choices[0].message.function_call
                            if isinstance(function_call.arguments, str):
                                chart_config = json.loads(function_call.arguments)
                            else:
                                chart_config = function_call.arguments
                        except Exception as e:
                            print(f"Error generating chart config: {e}")
                            chart_config = {
                                "title": {"text": "Data Visualization"},
                                "tooltip": {"trigger": "axis"},
                                "xAxis": {
                                    "type": "category", 
                                    "data": [str(row[cols[0]]) for row in sql_result['data'][:20]]
                                },
                                "yAxis": {"type": "value"},
                                "series": [{
                                    "name": cols[1],
                                    "type": "bar",
                                    "data": [row[cols[1]] for row in sql_result['data'][:20]]
                                }]
                            }

                # Prepare response
                assistant_content = "DATA_ANALYSIS_RESULT"
                metadata_to_save = {
                    "type": "analysis_result",
                    "user_query": message_data.content,
                    "sql_query": sql_query,
                    "summary": summary,
                    "data": {
                        "columns": sql_result.get("columns", []),
                        "rows": sql_result.get("data", []),
                        "total_rows": sql_result.get("row_count", 0)
                    },
                    "echartOption": chart_config
                }

            except Exception as sql_error:
                print(f"SQL execution failed: {sql_error}")
                assistant_content = f"I encountered an error while analyzing the data: {str(sql_error)}. Please try rephrasing your question or check if the data contains the information you're looking for."
        else:
            assistant_content = "I couldn't generate a valid analysis from your request. Could you please rephrase your question or be more specific about what you'd like to analyze?"

    # 6. Persist assistant's message
    assistant_message = Message(
        chat_session_id=session_id,
        role="assistant",
        content=assistant_content,
        metadata_=json.dumps(metadata_to_save) if metadata_to_save else None
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)

    return {
        "id": assistant_message.id,
        "role": assistant_message.role,
        "content": assistant_message.content,
        "created_at": assistant_message.created_at,
        "metadata": metadata_to_save
    }

async def generate_summary_with_context(client, user_request: str, sql_result: dict, chat_history: str) -> str:
    """Generate a contextual natural language summary"""
    
    total_rows = sql_result.get("row_count", 0)
    columns = sql_result.get("columns", [])
    data = sql_result.get("data", [])
    sample_data = data[:5] if data else []
    
    summary_prompt = f"""
Based on the conversation history and current analysis, provide a natural response.

CONVERSATION HISTORY:
{chat_history}

CURRENT USER REQUEST: "{user_request}"

ANALYSIS RESULTS:
- Total results: {total_rows}
- Columns: {', '.join(columns)}
- Sample data: {json.dumps(sample_data, default=str)}

Instructions:
1. Reference previous conversation context where relevant
2. Provide a conversational 2-3 sentence summary
3. Highlight key insights or patterns in the data
4. Use natural language, avoid technical jargon
5. If this builds on previous analysis, acknowledge that connection

Example: "Based on our previous discussion about sales trends, I found that..." or "Here's what the data shows about..."
"""

    try:
        summary_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3,
            max_tokens=300
        )
        return summary_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"I found {total_rows} results for your query about {user_request.lower()}."
