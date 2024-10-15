from leetcode.getQuestion import Query

try:
    question_id, question = Query(1)  # Fetch question with id 1
    print(f"Question {question_id}: {question.Title}")
    print(f"Content: {question.Content}")
    print(f"Topics: {question.Topics}")
except Exception as e:
    print(e)

