from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    id               = Column(Integer, primary_key=True, index=True)
    email            = Column(String, unique=True, nullable=False, index=True)
    name             = Column(String, nullable=True)
    oauth_provider   = Column(String, nullable=True)
    oauth_sub        = Column(String, nullable=True)
    picture          = Column(String, nullable=True)
    hashed_password  = Column(String, nullable=True)
    created_at       = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    chat_sessions    = relationship(
        "ChatSession",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    uploaded_files   = relationship(
        "UploadedFile",
        back_populates="user",
        cascade="all, delete-orphan"
    )

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id            = Column(String, primary_key=True, index=True)  # UUID
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False)
    title         = Column(String, nullable=False, default="New Chat")
    created_at    = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at    = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    is_active     = Column(Boolean, default=True, nullable=False)
    has_file      = Column(Boolean, default=False, nullable=False)
    # Relationships
    user          = relationship("User", back_populates="chat_sessions")
    messages      = relationship(
        "Message",
        back_populates="chat_session",
        cascade="all, delete-orphan"
    )
    uploaded_files = relationship(
        "UploadedFile",
        back_populates="chat_session",
        cascade="all, delete-orphan"
    )

# models.py

class Message(Base):
    __tablename__ = "messages"

    id              = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role            = Column(String, nullable=False)
    content         = Column(Text, nullable=False)
    
    # --- THIS IS THE FIX ---
    # Change the column type from Integer to JSON.
    # I've also renamed it to `metadata` for cleaner access in Pydantic.
    metadata_        = Column("metadata", JSON, nullable=True)
    # --- END OF FIX ---

    created_at      = Column(DateTime, server_default=func.now(), nullable=False)
    
    chat_session    = relationship("ChatSession", back_populates="messages")

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False)
    chat_session_id  = Column(String, ForeignKey("chat_sessions.id"), nullable=True)
    filename         = Column(String, nullable=False)
    original_filename= Column(String, nullable=False)
    file_path        = Column(String, nullable=False)
    file_size        = Column(Integer, nullable=False)
    mime_type        = Column(String, nullable=False)
    schema_info      = Column(JSON, nullable=True)
    upload_status    = Column(String, default="completed", nullable=False)
    created_at       = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    user             = relationship("User", back_populates="uploaded_files")
    chat_session     = relationship("ChatSession", back_populates="uploaded_files")
