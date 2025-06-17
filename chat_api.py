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
import httpx # <--- ADD aiohttp
from typing import Dict, Any, List
# You will need to pip install openai polars pydantic
# 1. Load .env
load_dotenv()
MCP_SERVER_URL = "http://localhost:8002"
# 2. Ensure tables exist
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
    class Config: orm_mode = True

class ChatSessionSchema(BaseModel):
    id: str; title: str; created_at: datetime; updated_at: datetime
    uploaded_files: List[UploadedFileSchema] = []
    class Config: orm_mode = True

class MessageResponseWithMetadata(BaseModel):
    id: int; role: str; content: str; created_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    class Config: orm_mode = True

class UploadResponseSchema(BaseModel):
    chat_session: ChatSessionSchema
    initial_message: MessageResponseWithMetadata

class ChatSessionCreate(BaseModel): title: str = "New Chat"
class MessageCreate(BaseModel): content: str
# Now, let's get the schemas for the OpenAI API
SummarizeDataArgs = create_model(
    'summarize_data',
    __doc__="Return a brief text summary of the dataset: #rows, #cols, missing values, and sample unique values.",
    # IMPORTANT: The file_locations argument is handled by the client now,
    # so we don't need to describe it to the LLM.
)
EchartArgs = create_model(
    'generate_echart_config',
    __doc__="Build an ECharts `option` dict from structured JSON data. Use this AFTER getting data from a SQL query.",
    data=(List[Dict[str, Any]], Field(..., description="The JSON data from a previous SQL query result.")),
    chart_type=(str, Field(..., description="e.g. 'bar' or 'pie'")),
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

@router.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
    authorization: str = Header(..., alias="Authorization"),  # Changed from Form to Header
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
    
    # Create a new chat session for this file
    chat_session = ChatSession(
        id=str(uuid.uuid4()),
        user_id=user.id,
        title=f"Analysis: {file.filename}"
    )
    db.add(chat_session)
    db.flush()  # Get the ID without committing
    
    # Save uploaded file with the new session ID
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
    db.refresh(chat_session)
    
    # Generate summary using MCP
    try:
        async with httpx.AsyncClient() as http_client:
            tool_response = await http_client.post(
                f"{MCP_SERVER_URL}/execute-tool",
                json={
                    "tool_name": "summarize_data",
                    "arguments": {"file_locations": [file_path]}
                },
                timeout=60.0
            )
            tool_response.raise_for_status()
            summary = tool_response.json()
    except Exception as e:
        summary = f"Error generating summary: {str(e)}"
    
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
            "uploaded_files": [
                {
                    "id": f.id,
                    "original_filename": f.original_filename
                } for f in session.uploaded_files
            ] if session.uploaded_files else []
        }
        for session in sessions
    ]

@router.post("/chat-sessions", response_model=ChatSessionSchema)
async def create_session(session_data: ChatSessionCreate, authorization: str = Header(...), db: Session = Depends(get_db)):
    user = get_current_user(authorization, db)
    session = ChatSession(id=str(uuid.uuid4()), user_id=user.id, title=session_data.title)
    db.add(session); db.commit(); db.refresh(session)
    return session

@router.post("/upload-csv", response_model=UploadResponseSchema)
async def upload_and_summarize(file: UploadFile = File(...), authorization: str = Header(...), db: Session = Depends(get_db)):
    user = get_current_user(authorization, db)
    if not file.filename.endswith('.csv'): raise HTTPException(400, "Only CSV files allowed")

    session = ChatSession(id=str(uuid.uuid4()), user_id=user.id, title=file.filename)
    db.add(session); db.flush()

    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as f: f.write(await file.read())
    
    uploaded_file = UploadedFile(user_id=user.id, chat_session_id=session.id, filename=os.path.basename(file_path), original_filename=file.filename, file_path=file_path, file_size=os.path.getsize(file_path))
    db.add(uploaded_file)
    
    summary = "Could not generate initial summary."
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{MCP_SERVER_URL}/execute-tool", json={"tool_name": "summarize_data", "arguments": {"file_locations": [file_path]}}, timeout=60)
            if resp.is_success: summary = resp.json()
    except Exception: pass

    welcome_content = f"I've analyzed **{file.filename}**. Here are the initial details:\n\n{summary}\n\nWhat would you like to explore further?"
    initial_message = Message(chat_session_id=session.id, role="assistant", content=welcome_content)
    db.add(initial_message)
    
    db.commit(); db.refresh(session); db.refresh(initial_message)
    
    return {"chat_session": session, "initial_message": initial_message}

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
            # parse out the stored JSON string into a real dict
            "metadata": (
                json.loads(msg.metadata_)
                if isinstance(msg.metadata_, str) and msg.metadata_ else
                msg.metadata_  # already a dict, or None
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
        assistant_content = "I can only analyze data if you upload a CSV file in this chat."
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

    # 3. Prepare history and file paths
    messages_history = db.query(Message)\
        .filter(Message.chat_session_id == session_id)\
        .order_by(Message.created_at.asc())\
        .all()
    file_paths = [f.file_path for f in session.uploaded_files]

    # 4. Build and send detailed prompt to OpenAI to generate SQL
    schema_info = session.uploaded_files[0].schema_info
    schema_description = json.dumps(schema_info, indent=2)

    prompt = f"""
    You have message history and file paths to work with:
    This is the context that you can work with to answer the user's question:
{messages_history}
For generic queries like hi, hello, help, etc., you can respond with a friendly message.
You are a SQL expert. Analyze the CSV data schema and convert user requests to SQL queries.

CSV Schema and Sample Data:
{schema_description}

Important Rules:
1. The table name is `csv_table`
2. Generate ONLY ONE SQL statement
3. Do NOT include semicolons at the end
4. For "top 10 products" type queries, use appropriate GROUP BY and ORDER BY
5. Column names are case-sensitive

User Request: "{message_data.content}"

Step 1: Check if this appears to be sales/business data (has columns like Product, Category, Quantity, Revenue, Sales, etc.)
Step 2: If not sales data, respond with "Sales Data: No"
Step 3: If sales data, generate a single SQL query to answer the user's request

Response Format:
Sales Data: Yes/No

SQL Query:
SELECT column1, column2 FROM csv_table WHERE conditions GROUP BY column1 ORDER BY column2 LIMIT 10
"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    response_content = response.choices[0].message.content.strip()
    print(f"OpenAI response: {response_content}")
    # 5. Handle OpenAI response
    metadata_to_save = None
    if response_content.startswith("Sales Data: No"):
        assistant_content = response_content
    else:
        # Extract SQL more carefully
        try:
            if "SQL Query:" in response_content:
                sql_section = response_content.split("SQL Query:")[1].strip()
                if sql_section.startswith("```sql"):
                    sql_section = sql_section[6:]
                if sql_section.startswith("```"):
                    sql_section = sql_section[3:]
                if sql_section.endswith("```"):
                    sql_section = sql_section[:-3]
                sql_query = sql_section.strip()
                if sql_query.endswith(';'):
                    sql_query = sql_query[:-1]
            else:
                sql_query = None
        except Exception as e:
            print(f"Error extracting SQL: {e}")
            sql_query = None

        if sql_query:
            try:
                # Execute SQL on CSV via MCP tool
                async with httpx.AsyncClient() as http_client:
                    exec_resp = await http_client.post(
                        f"{MCP_SERVER_URL}/execute-tool",
                        json={
                            "tool_name": "execute_sql_query",
                            "arguments": {
                                "file_locations": file_paths,
                                "sql_query": sql_query
                            }
                        },
                        timeout=60.0
                    )
                    exec_resp.raise_for_status()
                    sql_result = exec_resp.json()

                    # Generate chart from results if we have data
                    chart_config = None
                    if sql_result.get("data") and len(sql_result["data"]) > 0:
                        cols = sql_result["columns"]
                        print(f"SQL Result Columns: {cols}")
                        if len(cols) >= 2:
                            try:
                                chart_resp = await http_client.post(
                                    f"{MCP_SERVER_URL}/execute-tool",
                                    json={
                                        "tool_name": "generate_echart_config",
                                        "arguments": {
                                            "file_locations": file_paths,
                                            "chart_type": "bar",
                                            "x_axis": cols[0],
                                            "y_axis": cols[1]
                                        }
                                    },
                                    timeout=60.0
                                )
                                chart_resp.raise_for_status()
                                chart_config = chart_resp.json()
                                print(f"Generated chart config: {chart_config}")
                            except Exception as chart_error:
                                print(f"Chart generation failed: {chart_error}")
                                chart_config = None

                    # Generate natural language summary using OpenAI
                    summary = await generate_summary(client, message_data.content, sql_result)

                    # Flag for frontend and full metadata
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
                assistant_content = f"Sorry, there was an error executing the SQL query: {str(sql_error)}"
        else:
            assistant_content = "I couldn't generate a valid SQL query from your request. Please try rephrasing your question."

    # 6. Persist assistant's message
    assistant_message = Message(
        chat_session_id=session_id,
        role="assistant",
        content=assistant_content,
        metadata_=metadata_to_save
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


async def generate_summary(client, user_request: str, sql_result: dict) -> str:
    """Generate a concise natural language summary"""
    
    total_rows = sql_result.get("row_count", 0)
    columns = sql_result.get("columns", [])
    data = sql_result.get("data", [])
    sample_data = data[:5] if data else []
    
    summary_prompt = f"""
Analyze this data query result and provide a brief, conversational summary.

User asked: "{user_request}"
Total results: {total_rows}
Columns: {', '.join(columns)}
Sample data: {json.dumps(sample_data, default=str)}

Provide a 2-3 sentence summary that:
1. Directly answers the user's question
2. Highlights key insights or patterns
3. Uses conversational language

Do not include technical details or mention SQL.
"""

    try:
        summary_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return summary_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Found {total_rows} results for your query about {user_request.lower()}."

# New endpoint to associate uploaded file with chat session
@router.put("/uploaded-files/{file_id}/associate-session")
async def associate_file_with_session(
    file_id: int,
    chat_session_id: str,
    authorization: str = Header(..., alias="Authorization"),
    db: Session = Depends(get_db)
):
    user = get_current_user(authorization, db)
    
    # Verify file belongs to user
    uploaded_file = db.query(UploadedFile).filter(
        UploadedFile.id == file_id,
        UploadedFile.user_id == user.id
    ).first()
    
    if not uploaded_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Verify session belongs to user
    session = db.query(ChatSession).filter(
        ChatSession.id == chat_session_id,
        ChatSession.user_id == user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Associate file with session
    uploaded_file.chat_session_id = chat_session_id
    db.commit()
    
    return {"message": "File associated with chat session successfully"}