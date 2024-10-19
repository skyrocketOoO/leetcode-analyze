# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
from typing import List
import matplotlib.pyplot as plt
import json


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