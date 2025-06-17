from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
import os
from auth import router as auth_router
load_dotenv()
from chat_api import router as chat_router

# Add after your existing router includes


# … import other routers …


# Needed for Authlib OAuth state (only for OAuth routes)

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

# after you instantiate app = FastAPI()
origins = [
    "http://localhost:3000",   # your React dev server
    # add any other front-end hosts here (e.g. production URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # or ["*"] in dev
    allow_credentials=True,
    allow_methods=["*"],              # ← THIS allows OPTIONS, POST, GET, etc.
    allow_headers=["*"],              # ← allows Content-Type, Authorization, etc.
)

# Session cookie for OAuth state
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY")
)

app.include_router(auth_router)
app.include_router(chat_router)