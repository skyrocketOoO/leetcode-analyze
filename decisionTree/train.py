# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
from typing import List
import matplotlib.pyplot as plt


def Train(features: List, targetLabels: List, feature_names: List[str] = None, class_names: List[str] = None):
  # Step 2: Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(features, targetLabels, test_size=0.05, random_state=42)

  # Step 3: Initialize and train the Decision Tree Classifier
  clf = DecisionTreeClassifier(random_state=42)
  clf.fit(X_train, y_train)

  # Step 4: Make predictions on the test set
  y_pred = clf.predict(X_test)

  # Step 5: Evaluate the model's performance
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy}")
  print("Classification Report:")
  print(classification_report(y_test, y_pred))

  # Step 6: Visualize the decision tree
  plt.figure(figsize=(12,8))
  tree.plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
  plt.show()

if __name__ == "__main__":
  Train()