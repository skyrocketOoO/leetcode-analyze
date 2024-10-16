from leetcode.getQuestion import Question, Query
from typing import List, Dict
import json
import os
import time

DATA_FILE = "questions.json"

def load_local_questions() -> Dict[int, Question]:
    """Load all questions from a local JSON file if it exists."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            data = json.load(file)
            # Deserialize into Question objects
            return {int(k): Question(**v) for k, v in data.items()}
    return {}

def save_local_questions(questions: Dict[int, Question]):
    """Save all questions to a single local JSON file."""
    with open(DATA_FILE, "w") as file:
        # Serialize Question objects into their dictionaries
        json.dump({qid: question.__dict__ for qid, question in questions.items()}, file)

def GetAllQuestions() -> List[Question]:
    """Fetch all Leetcode questions and store them locally in a single JSON file."""
    questions = load_local_questions()  # Load existing questions from file
    total, _ = Query(1)  # Get the total number of questions

    # Start fetching and saving questions
    for i in range(1, total+1):
        if i not in questions:
            print(f"query {i}")
            # If the question isn't stored locally, fetch it from the API
            _, question = Query(i)
            questions[i] = question  # Add it to the dictionary
            time.sleep(0.4)  # Rate-limiting sleep

    save_local_questions(questions)  # Save all questions into one file
    return list(questions.values())  # Return the list of Question objects

if __name__ == "__main__":
    questions = GetAllQuestions()
    print(f"Total questions fetched: {len(questions)}")