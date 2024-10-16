from leetcode.getAllQuestions import GetAllQuestions
from decisionTree.train import Train
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.cleanHtmlContent import CleanHtmlContent
from datetime import datetime
import numpy as np
import time
import os, json

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
    if question.Content is None or question.Category != "Algorithms":
        continue
    algQuestions.append(question)

print("transform topics")
targetLabels = []
for question in algQuestions:
    targetLabels.append(question.Topics)
mlb = MultiLabelBinarizer()
targetLabels = mlb.fit_transform(targetLabels)

class_counts = np.sum(targetLabels, axis=0)
# Get the class names
class_names = mlb.classes_
class_count_dict = {class_name: int(count) for class_name, count in zip(class_names, class_counts)}

# Write the dictionary to a JSON file
with open(f"{recordPath}/class_counts.json", 'w') as json_file:
    json.dump(class_count_dict, json_file, indent=4)

print("transform features")
features = []
for question in algQuestions:
    if question.Content is None or question.Category != "Algorithms":
        continue
    features.append("Title: " + question.Title + "\n" + question.Content)
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(features)

print("training")
Train(features, targetLabels, vectorizer.get_feature_names_out(), class_names, recordPath)
    
executionTime = round(time.time()-startTime, 2)
print(f"Execution time: {executionTime} seconds")

with open(f"{recordPath}/report.json", 'r') as f:
    data = json.load(f)
    data["total questions"] = len(questions)
    data["total algorithms"] = len(algQuestions)
    data["execution time"] = executionTime
with open(f"{recordPath}/report.json", 'w') as f:
    json.dump(data, f, indent=4)
    