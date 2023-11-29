from models import Question
from sqlalchemy.orm import Session

# 데이터를 처리하는 함수
def get_question_list(db: Session):
    question_list = db.query(Question)\
        .order_by(Question.create_date.desc())\
        .all()
    
    return question_list