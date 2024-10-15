import requests
import json
from typing import Tuple, Any, List
from dataclasses import dataclass

@dataclass
class Question:
  Id: int
  Title: str
  Content: str
  Topics: List[str]

def Query(id: int) -> Tuple[int, Question]:
  url = "https://leetcode.com/graphql/"
  query = '''
  query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
    problemsetQuestionList: questionList(
      categorySlug: $categorySlug
      limit: $limit
      skip: $skip
      filters: $filters
    ) {
      total: totalNum
      questions: data {
        frontendQuestionId: questionFrontendId
        title
        content
        topicTags {
          name
        }
      }
    }
  }
  '''
  
  variables = {
      "categorySlug": "",
      "skip": id-1,
      "limit": 1,
      "filters": {}
  }

  # Create the JSON payload with the query and variables
  payload = {
      "query": query,
      "variables": variables
  }
  headers = {
      "Content-Type": "application/json",
  }

  response = requests.post(url, headers=headers, json=payload)

  if response.status_code == 200:
      data = response.json()
      problemset = data.get("data", {}).get("problemsetQuestionList", {})
      questions = problemset.get("questions", [])
      if questions:
        return problemset['total'], parseQuestion(questions[0])
        # print(json.dumps(questions, indent=2))  # Print the questions in a readable format
      else:
        raise Exception("No question found for the given ID")
  else:
    raise Exception(f"Query failed with status code {response.status_code}: {response.text}")

def parseQuestion(data: any) -> Question:
  return Question(
    Id=int(data['frontendQuestionId']),
    Title=data['title'],
    Content=data['content'],
    Topics=[tag['name'] for tag in data['topicTags']]
)