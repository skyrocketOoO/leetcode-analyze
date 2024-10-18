from leetcode.getAllQuestions import GetAllQuestions
from decisionTree.train import Train
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.cleanHtmlContent import CleanHtmlContent
from scipy.sparse import vstack as sparse_vstack
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
# Prepare feature and target labels
features, targetLabels = [], []
for question in questions:
    if question.Content is None or question.Category != "Algorithms":
        continue
    targetLabels.append(question.Topics)
    features.append("Title: " + question.Title + "\n" + question.Content)

print("Transforming topics")
mlb = MultiLabelBinarizer()
targetLabels = mlb.fit_transform(targetLabels)

print("Transforming features")
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(features)  # This will be a sparse matrix

# Get the class names and counts
class_counts = np.sum(targetLabels, axis=0)
class_names = mlb.classes_
class_count_dict = {class_name: int(count) for class_name, count in zip(class_names, class_counts)}

# Oversample on less represented classes
class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
index_to_class = {index: class_name for index, class_name in enumerate(class_names)}

while True:
    # Find the class with the minimum count
    minClassname, minCount = None, None
    for cls, c in class_count_dict.items():
        if minCount is None or minCount > c:
            minClassname = cls
            minCount = c

    # Stop if the minimum class count is greater than 10
    if minCount > 10:
        break

    # Find the data points belonging to the minority class
    indexed = []
    for i, labels in enumerate(targetLabels):
        if labels[class_to_index[minClassname]] == 1:  # Check for one-hot encoding match
            indexed.append(i)

    # Perform oversampling by appending the minority class data points
    for ind in indexed:
        for i, v in enumerate(targetLabels[ind]):  # Enumerating over each one-hot encoded label array
            if v == 1:  # If the value is 1 (indicating the class), increment its count
                class_name = index_to_class[i]
                class_count_dict[class_name] += 1

        # Append the features and target labels properly using sparse_vstack for sparse arrays
        features = sparse_vstack([features, features[ind]])  # Use sparse_vstack for sparse matrices
        targetLabels = np.vstack([targetLabels, targetLabels[ind]])  # Append target labels as rows


# Write the dictionary to a JSON file
with open(f"{recordPath}/class_counts.json", 'w') as json_file:
    json.dump(class_count_dict, json_file, indent=4)

print("training")
Train(features, targetLabels, vectorizer.get_feature_names_out(), class_names, recordPath)
    
executionTime = round(time.time()-startTime, 2)
print(f"Execution time: {executionTime} seconds")

with open(f"{recordPath}/report.json", 'r') as f:
    data = json.load(f)
    data["total questions"] = len(questions)
    data["total algorithms"] = len(targetLabels)
    data["execution time"] = executionTime
with open(f"{recordPath}/report.json", 'w') as f:
    json.dump(data, f, indent=4)
    