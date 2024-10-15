from leetcode.getQuestion import Question, Query
from typing import List
import json
import os
import time

DATA_DIR = "leetcode_questions"

def load_local_question(question_id: int) -> Question:
    """Load a single question from local storage if the file exists."""
    filepath = os.path.join(DATA_DIR, f"{question_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            data = json.load(file)
            return Question(**data)
    return None

def save_local_question(question: Question):
    """Save a single question to local storage."""
    os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the directory exists
    filepath = os.path.join(DATA_DIR, f"{question.Id}.json")
    with open(filepath, "w") as file:
        json.dump(question.__dict__, file)

def GetAllQuestions() -> List[Question]:
    """Fetch all Leetcode questions and store them locally as separate files."""
    questions = []
    total, _ = Query(1)  # Get the total number of questions

    # Start fetching and saving questions
    for i in range(1, total+1):
        # Check if the question already exists locally
        question = load_local_question(i)
        if question is None:
            print(f"query {i}")
            # If the question isn't stored locally, fetch it from the API
            _, question = Query(i)
            save_local_question(question)  # Save it to a separate file
        questions.append(question)
        time.sleep(0.2)  # Rate-limiting sleep
    
    return questions

if __name__ == "__main__":
    questions = GetAllQuestions()
    print(f"Total questions fetched: {len(questions)}")
