from leetcode.getAllQuestions import GetAllQuestions
from decisionTree.train import Train
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.cleanHtmlContent import CleanHtmlContent

try:
    questions = GetAllQuestions()
    
    # transform topics
    targetLabels = []
    for question in questions:
        targetLabels.append(question.Topics)
    mlb = MultiLabelBinarizer()
    targetLabels = mlb.fit_transform(targetLabels)
    
    # transform features
    features = []
    for question in questions:
        features.append("Title: " + question.Title + "\n" + CleanHtmlContent(question.Content))
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(features)
    Train(features, targetLabels, vectorizer.get_feature_names_out, mlb.classes_)
        
except Exception as e:
    print(e)

