from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from constants import FEEDBACK_SQL_PATH


engine = create_engine(FEEDBACK_SQL_PATH)
Session = sessionmaker(
    bind=engine,
)

