from graphql.queryLeetcodeQuestion import Question, Query
from typing import List
import time

def GetAllLeetcodeQuestions() -> List[Question]:
  total, _ = Query(1)
  questions = []
  for i in range(1, total+1):
    _, question = Query(i)
    questions.append(question)
    time.sleep(0.2)
  return questions