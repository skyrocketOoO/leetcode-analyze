from leetcode.getAllQuestions import GetAllQuestions
from decisionTree.train import Train
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.cleanHtmlContent import CleanHtmlContent
import numpy as np
import time

startTime = time.time()

print("get all questions")
questions = GetAllQuestions()

print("transform topics")
# transform topics
targetLabels = []
for question in questions:
    if question.Content is None or question.Category != "Algorithms":
        continue
    targetLabels.append(question.Topics)
mlb = MultiLabelBinarizer()
targetLabels = mlb.fit_transform(targetLabels)
class_counts = np.sum(targetLabels, axis=0)

# Get the class names
class_names = mlb.classes_

# Print class counts for each class
for class_name, count in zip(class_names, class_counts):
    print(f"{class_name}: {count}")

print("transform features")
# transform features
features = []
for question in questions:
    if question.Content is None or question.Category != "Algorithms":
        continue
    features.append("Title: " + question.Title + "\n" + question.Content)
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(features)

print(len(targetLabels))
print("training")
# print(features is None, targetLabels is None, )
Train(features, targetLabels, vectorizer.get_feature_names_out(), mlb.classes_)
    
print(f"Execution time: {time.time()-startTime} seconds")

