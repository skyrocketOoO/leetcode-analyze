# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.cleanHtmlContent import CleanHtmlContent
from scipy.sparse import vstack as sparse_vstack
from sklearn.model_selection import train_test_split
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import scipy
import json

def PreProcessing(algQuestions: List, recordPath: str, overSample: bool, KlBoundary: float):
  # Prepare feature and target labels
  features, targetLabels = [], []
  for question in algQuestions:
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

  # Write the class count dictionary to a JSON file
  with open(f"{recordPath}/class_counts.json", 'w') as json_file:
      json.dump(class_count_dict, json_file, indent=4)

  # Split the dataset
  X_train, X_test, y_train, y_test = train_test_split(features, targetLabels, test_size=0.05, random_state=42)

  overCount = 0
  nClass = len(class_count_dict.keys())
  uniDis = [1 / nClass for _ in range(nClass)]

  while overSample:
    totalC = sum(class_count_dict.values())
    curDis = [v / totalC for v in class_count_dict.values()]
    kl_divergence = scipy.stats.entropy(curDis, uniDis)
    print(kl_divergence)
    if kl_divergence < KlBoundary:
      break
    
    # Find the class with the minimum count
    minClassname = min(class_count_dict, key=class_count_dict.get)
    minCount = class_count_dict[minClassname]

    # Find the data points belonging to the minority class
    indexed = [i for i, labels in enumerate(y_train) if labels[class_to_index[minClassname]] == 1]

    if not indexed:
      break

    # Find the least affected index
    lessIndex = min(indexed, key=lambda idx: np.sum(y_train[idx]))

    # Update the class counts and perform oversampling
    for i, v in enumerate(y_train[lessIndex]):
      if v == 1:
        class_count_dict[index_to_class[i]] += 1

    # Append the features and target labels properly using sparse_vstack for sparse arrays
    X_train = sparse_vstack([X_train, X_train[lessIndex]])  # Use sparse_vstack for sparse matrices
    y_train = np.vstack([y_train, y_train[lessIndex]])  # Append target labels as rows

    overCount += 1

  return X_train, X_test, y_train, y_test, vectorizer, mlb, overCount


def Train(X_train: List, y_train: List, X_test: List, y_test: List, feature_names: List[str], class_names: List[str], recordPath: str):
  # Initialize and train the Decision Tree Classifier on the resampled data
  clf = DecisionTreeClassifier(random_state=42)
  clf.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = clf.predict(X_test)

  # Evaluate the model's performance
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy}")
  
  # Store report
  with open(f"{recordPath}/report.json", 'r') as f:
    data = json.load(f)
    data["accuracy_score"] = accuracy
  with open(f"{recordPath}/report.json", 'w') as f:
    json.dump(data, f, indent=4)
    
  report_dict = classification_report(y_test, y_pred, output_dict=True)
  with open(f"{recordPath}/classification_report.json", 'w') as f:
    json.dump(report_dict, f, indent=4)

  # plt.figure(figsize=(12,8))
  # tree.plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
  # plt.savefig(f"{recordPath}/decision_tree.png")

if __name__ == "__main__":
  Train()