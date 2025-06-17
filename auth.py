import os
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.middleware.sessions import SessionMiddleware
from jose import jwt
from sqlalchemy.orm import Session
from database import SessionLocal, Base, engine
import models
from dotenv import load_dotenv

# 1. Load .env
load_dotenv()

# 2. Ensure tables exist
Base.metadata.create_all(bind=engine)

# 3. Set up router and OAuth client
router = APIRouter()
oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt

from models import User
from pydantic import BaseModel
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
router = APIRouter()

class UserCreate(BaseModel):
    email: str
    password: str
    name: str

class UserLogin(BaseModel):
    email: str
    password: str

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=1))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@router.post("/signup")
def signup(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter_by(email=payload.email).first():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email already registered")
    hashed = pwd_ctx.hash(payload.password)
    user = User(
        email=payload.email,
        name=payload.name,
        hashed_password=hashed,
        oauth_provider=None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": user.email})
    return {
        "access_token": token, 
        "token_type": "bearer",
        "user": {
            "name": user.name,
            "email": user.email
        }
    }

@router.post("/signin")
def signin(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=payload.email).first()
    if not user or not user.hashed_password or not pwd_ctx.verify(payload.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    token = create_access_token({"sub": user.email})
    return {
        "access_token": token, 
        "token_type": "bearer",
        "user": {
            "name": user.name,
            "email": user.email
        }
    }

@router.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth_callback")
    # Add prompt=select_account to force account chooser
    return await oauth.google.authorize_redirect(
        request, 
        redirect_uri, 
        prompt="select_account"
    )

@router.get("/auth/callback")
async def auth_callback(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as err:
        raise HTTPException(400, f"OAuth error: {err}")

    # Grab userinfo
    userinfo = token.get("userinfo")
    if not userinfo:
        resp = await oauth.google.get("userinfo", token=token)
        userinfo = resp.json()

    sub     = userinfo["sub"]
    email   = userinfo["email"]
    name    = userinfo.get("name")
    picture = userinfo.get("picture")

    # Upsert user
    user = db.query(models.User).filter_by(
        oauth_provider="google", oauth_sub=sub
    ).first()
    if not user:
        user = models.User(
            oauth_provider="google",
            oauth_sub=sub,
            email=email,
            name=name,
            picture=picture,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # Issue JWT
    SECRET_KEY = os.getenv("SECRET_KEY")
    access_token = jwt.encode({"sub": sub}, SECRET_KEY, algorithm="HS256")
    
    # Store session info for logout
    request.session["oauth_provider"] = "google"
    
    redirect_url = f"{os.getenv('FRONTEND_URL')}/auth/callback#token={access_token}"
    return RedirectResponse(redirect_url)



@router.post("/logout")
async def logout(request: Request):
    # Clear any session data
    request.session.clear()
    return {"message": "Logged out successfully"}

@router.get("/me")
async def get_current_user(request: Request, db: Session = Depends(get_db)):
    # Get token from Authorization header
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No valid token")
    
    token = auth_header.split(" ")[1]
    
    try:
        # Decode JWT to get user identifier
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        
        # For OAuth users, sub is the oauth_sub
        user = db.query(User).filter(
            (User.oauth_sub == user_id) | (User.email == user_id)
        ).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        return {
            "name": user.name,
            "email": user.email,
            "picture": user.picture,
            "oauth_provider": user.oauth_provider  # Add this line
        }
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")