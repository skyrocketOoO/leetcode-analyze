from leetcode.getAllQuestions import GetAllQuestions
from decisionTree.train import Train, PreProcessing
from datetime import datetime
import time
import os, json

OVER_SAMPLE = True
KL_BOUNDARY = 0.8

startTime = time.time()

currentTime = datetime.now()
formattedTime = currentTime.strftime('%Y-%m-%d:%H-%M-%S')
recordPath = f"records/{formattedTime}"
os.mkdir(recordPath)
with open(f"{recordPath}/report.json", 'w') as f:
    json.dump({}, f, indent=4)

print("get all questions")
questions = GetAllQuestions()

print("Remove non-algorithm data")
algQuestions = []
for question in questions:
    if question.Content is None or question.Category!= "Algorithms":
        continue
    algQuestions.append(question)

X_train, X_test, y_train, y_test, vectorizer, mlb, overCount = PreProcessing(algQuestions, recordPath, OVER_SAMPLE, KL_BOUNDARY)

print("training")
Train(X_train, y_train, X_test, y_test, vectorizer.get_feature_names_out(), mlb.classes_, recordPath)
    
executionTime = round(time.time()-startTime, 2)
print(f"Execution time: {executionTime} seconds")

with open(f"{recordPath}/report.json", 'r') as f:
    data = json.load(f)
    data["total questions"] = len(questions)
    data["total algorithms"] = len(algQuestions)
    data["overSampling"] = overCount
    data["execution time"] = executionTime
with open(f"{recordPath}/report.json", 'w') as f:
    json.dump(data, f, indent=4)
    