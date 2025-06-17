# reset_db.py
import logging
from database import engine, Base
# ensure all of your models are imported so Base.metadata knows about them:
import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_database():
    logger.info("Dropping all existing tables...")
    Base.metadata.drop_all(bind=engine)

    logger.info("Creating tables from models.py...")
    Base.metadata.create_all(bind=engine)

    logger.info("Database reset complete.")

if __name__ == "__main__":
    reset_database()
