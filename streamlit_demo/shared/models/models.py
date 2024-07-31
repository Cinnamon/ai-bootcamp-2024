from sqlalchemy import Column, Integer, JSON, String
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Feedback(Base):
    __tablename__ = 'Feedback'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String)
    data = Column(JSON, default=dict())

