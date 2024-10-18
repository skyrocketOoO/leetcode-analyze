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

OVER_SAMPLE = True

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

import scipy.stats
nClass = len(class_count_dict.keys())
uniDis = [1 / nClass for _ in range(nClass)]

while OVER_SAMPLE:
    totalC = sum(class_count_dict.values())
    curDis = [v / totalC for v in class_count_dict.values()]
    kl_divergence = scipy.stats.entropy(curDis, uniDis)
    print(kl_divergence)
    if kl_divergence < 0.1:
        break
    
    # Find the class with the minimum count
    minClassname, minCount = None, None
    for cls, c in class_count_dict.items():
        if minCount is None or minCount > c:
            minClassname = cls
            minCount = c
    
    # Find the data points belonging to the minority class
    indexed = []
    for i, labels in enumerate(targetLabels):
        if labels[class_to_index[minClassname]] == 1:  # Check for one-hot encoding match
            indexed.append(i)

    # Perform oversampling by appending the minority class data points
    # Find the less affect index
    lessIndex, lessAffectClass = None, float('inf')
    for ind in indexed:
        c = 0
        for v in targetLabels[ind]:
            c += v
        if c < lessAffectClass:
            lessIndex = ind
    
    for i, v in enumerate(targetLabels[lessIndex]):  # Enumerating over each one-hot encoded label array
        class_name = index_to_class[i]
        class_count_dict[class_name] += 1

    # Append the features and target labels properly using sparse_vstack for sparse arrays
    features = sparse_vstack([features, features[lessIndex]])  # Use sparse_vstack for sparse matrices
    targetLabels = np.vstack([targetLabels, targetLabels[lessIndex]])  # Append target labels as rows


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
    