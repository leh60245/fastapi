from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from database import Base   # database.py에서 정의한 Base.

# (임시) 이미지가 들어왔을 때 저장할 형식을 임시로 구상했다.
class Question(Base):
    __tablename__ = "question"  # 모델에 의해 관리되는 테이블의 이름

    id = Column(Integer, primary_key=True)
    subject = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    create_date = Column(DateTime, nullable=False)
    
    img = Column(Text, primary_key=False)   # 저장할 이미지 데이터

# (임시) 이미지에 대한 결과를 돌려주는 형식
class Answer(Base):
    __tablename__ = "answer"

    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    create_date = Column(DateTime, nullable=False)
    question_id = Column(Integer, ForeignKey("question.id"))    # 답변을 질문과 연결하기 위해 추가한 속성
    question = relationship("Question", backref="answers")      # 두 번째 backref 파라미터는 역참조 설정으로 쉽게 말해 질문에서 답변을 거꾸로 참조하는 것을 의미
    
    key_point = Column(Text, nullable=False)    # key point를 넘겨준다.