# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
from typing import List
import matplotlib.pyplot as plt
import json


def Train(features: List, targetLabels: List, feature_names: List[str], class_names: List[str], recordPath: str):
  # Step 2: Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(features, targetLabels, test_size=0.05, random_state=42)

  # Step 3: Initialize and train the Decision Tree Classifier
  clf = DecisionTreeClassifier(random_state=42)
  clf.fit(X_train, y_train)

  # Step 4: Make predictions on the test set
  y_pred = clf.predict(X_test)

  # Step 5: Evaluate the model's performance
  accuracy = accuracy_score(y_test, y_pred)
  with open(f"{recordPath}/report.json", 'r') as f:
    data = json.load(f)
    data["accuracy_score"] = accuracy
  with open(f"{recordPath}/report.json", 'w') as f:
    json.dump(data, f, indent=4)
    
  print(f"Accuracy: {accuracy}")
  print("Classification Report:")
  report_dict = classification_report(y_test, y_pred, output_dict=True)
  # print(classification_report(y_test, y_pred))
  with open(f"{recordPath}/classification_report.json", 'w') as f:
    json.dump(report_dict, f, indent=4)

  # Step 6: Visualize the decision tree
  plt.figure(figsize=(12,8))
  tree.plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
  plt.savefig(f"{recordPath}/decision_tree.png")

if __name__ == "__main__":
  Train()