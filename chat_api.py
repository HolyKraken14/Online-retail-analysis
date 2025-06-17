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

class ChatSessionCreate(BaseModel):
    title: str = "New Chat"

class MessageCreate(BaseModel):
    content: str
    role: str  # 'user' or 'assistant'

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
class MessageResponseWithMetadata(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

# Now, let's get the schemas for the OpenAI API
SummarizeDataArgs = create_model(
    'summarize_data',
    __doc__="Return a brief text summary of the dataset: #rows, #cols, missing values, and sample unique values.",
    # IMPORTANT: The file_locations argument is handled by the client now,
    # so we don't need to describe it to the LLM.
)
GenerateEchartConfigArgs = create_model(
    'generate_echart_config',
    __doc__="Build an ECharts `option` dict for common charts like line, bar, or pie.",
    chart_type=(str, Field(..., description="e.g. 'line', 'bar', 'pie'")),
    x_axis=(Optional[str], Field(None, description="Column for X axis.")),
    y_axes=(Optional[List[str]], Field(None, description="Column(s) for Y axis.")),
    filters=(Optional[str], Field(None, description="Optional Pandas query string to filter data before plotting."))
)


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

@router.post("/chat-sessions")
async def create_chat_session(
    session_data: ChatSessionCreate,
    authorization: str = Header(..., alias="Authorization"),
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
    
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at
    }

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
    
    messages = db.query(Message).filter(
        Message.chat_session_id == session_id
    ).order_by(Message.created_at.asc()).all()
    
    return [
        {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at
        }
        for message in messages
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

    history_for_llm = [
        {"role": m.role, "content": m.content}
        for m in messages_history
    ] + [{"role": "user", "content": message_data.content}]

    # 4. Build and send detailed prompt to OpenAI to generate SQL
    schema_info = session.uploaded_files[0].schema_info
    schema_description = json.dumps(schema_info, indent=2)

    prompt = f"""
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

    # 5. Handle OpenAI response
    metadata_to_save = None
    if response_content.startswith("Sales Data: No"):
        assistant_content = response_content
    else:
        # Extract SQL more carefully
        try:
            # Look for SQL Query: section
            if "SQL Query:" in response_content:
                sql_section = response_content.split("SQL Query:")[1].strip()
                
                # Remove code block markers if present
                if sql_section.startswith("```sql"):
                    sql_section = sql_section[6:]  # Remove ```sql
                if sql_section.startswith("```"):
                    sql_section = sql_section[3:]   # Remove ```
                if sql_section.endswith("```"):
                    sql_section = sql_section[:-3]  # Remove trailing ```
                
                # Clean up the query
                sql_query = sql_section.strip()
                
                # Remove any trailing semicolons
                if sql_query.endswith(';'):
                    sql_query = sql_query[:-1]
                
                print(f"Extracted SQL Query: {repr(sql_query)}")
                
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
                    if sql_result.get("data") and len(sql_result["data"]) > 0:
                        cols = sql_result["columns"]
                        if len(cols) >= 2:  # Need at least 2 columns for a chart
                            try:
                                chart_resp = await http_client.post(
                                    f"{MCP_SERVER_URL}/execute-tool",
                                    json={
                                        "tool_name": "generate_echart_config",
                                        "arguments": {
                                            "file_locations": file_paths,
                                            "chart_type": "bar",
                                            "x_axis": cols[0],
                                            "y_axes": cols[1:2]  # Take only the first numeric column
                                        }
                                    },
                                    timeout=60.0
                                )
                                chart_resp.raise_for_status()
                                metadata_to_save = chart_resp.json()
                            except Exception as chart_error:
                                print(f"Chart generation failed: {chart_error}")
                                metadata_to_save = None

                    # Format response
                    total_rows = sql_result.get("row_count", 0)
                    sample_data = sql_result['data'][:5] if sql_result.get('data') else []
                    
                    assistant_content = (
                        f"Here is the analysis for: **{message_data.content}**\n\n"
                        f"**SQL Query Executed:**\n```sql\n{sql_query}\n```\n\n"
                        f"**Results:** {total_rows} rows found\n\n"
                        f"**Sample Data:**\n```json\n{json.dumps(sample_data, indent=2, default=str)}\n```"
                    )
                    
            except Exception as sql_error:
                print(f"SQL execution failed: {sql_error}")
                assistant_content = f"Sorry, there was an error executing the SQL query: {str(sql_error)}"
                
        else:
            assistant_content = "I couldn't generate a valid SQL query from your request. Please try rephrasing your question."

    # 6. Persist assistant’s message
    assistant_message = Message(
        chat_session_id=session_id,
        role="assistant",
        content=assistant_content,
        metadata_={"echartOption": metadata_to_save} if metadata_to_save else None
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)

    return {
        "id": assistant_message.id,
        "role": assistant_message.role,
        "content": assistant_message.content,
        "created_at": assistant_message.created_at,
        "metadata": {"echartOption": metadata_to_save} if metadata_to_save else None
}

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