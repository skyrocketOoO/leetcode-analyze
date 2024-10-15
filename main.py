from leetcode.getAllQuestions import GetAllQuestions
from decisionTree.train import Train
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.cleanHtmlContent import CleanHtmlContent


print("get all questions")
questions = GetAllQuestions()

print("transform topics")
# transform topics
targetLabels = []
for question in questions:
    if question.Content is None:
        continue
    targetLabels.append(question.Topics)
mlb = MultiLabelBinarizer()
targetLabels = mlb.fit_transform(targetLabels)
print("transform features")
# transform features
features = []
for question in questions:
    if question.Content is None:
        continue
    features.append("Title: " + question.Title + "\n" + question.Content)
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(features)

print(len(targetLabels))
print("training")
# print(features is None, targetLabels is None, )
Train(features, targetLabels, vectorizer.get_feature_names_out(), mlb.classes_)
        


