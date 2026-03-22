import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.preprocessing import FunctionTransformer
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_recall_curve
)
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv(r"D:\Complete_proj\Email_spam\model\data\email.csv")
print(f"shape of data : {data.shape}")

mail=data['Message'].astype(str)
target=data['Category']
x_train,x_test,y_train,y_test=train_test_split(mail,target,test_size=0.2,shuffle=True,random_state=42)


mask=y_train.isin(['ham','spam'])
x_train=x_train[mask]
y_train=y_train[mask]

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# def transforms_text(text):
#     # Lowercase and tokenize
#     tokens = nltk.word_tokenize(text.lower())
    
#     transformed = []
#     for word in tokens:
#         # Remove standalone punctuation and stopwords
#         if word not in stop_words and word not in punctuation:
#             # Check if it has at least one alphanumeric character 
#             # (Allows "w1nn3r" but removes "!!!")
#             if any(char.isalnum() for char in word):
#                 transformed.append(ps.stem(word))
    
#     return " ".join(transformed)

def transforms_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase and tokenize
    tokens = nltk.word_tokenize(text.lower())
    
    transformed = []
    for word in tokens:
        # Keep words, keep numbers, and keep currency symbols
        # We only remove punctuation that is NOT a currency sign or number
        if word not in stop_words and (word.isalnum() or word in ['$']):
            # Optional: Remove stemming. 
            # Sometimes stemming "winning" to "win" loses the "urgency" of spam.
            # Let's keep stemming for now but ensure it doesn't kill the word.
            transformed.append(ps.stem(word))
    
    return " ".join(transformed)

def text_cleaner_wrapper(text_input):
    """
    Processes a list or Series of strings one by one to avoid 
    the massive memory allocation spike of np.vectorize.
    """
    # If the input is a single string (from a single prediction), 
    # wrap it in a list so the loop works.
    if isinstance(text_input, str):
        text_input = [text_input]
        
    # Standard Python list comprehension: low memory overhead
    return [transforms_text(text) for text in text_input]

def full_pipeline():

    pipeline = Pipeline([
        # ('cleaner', FunctionTransformer(text_cleaner_wrapper)),
        ('tfidf', TfidfVectorizer()),
        ('rnd_clf', RandomForestClassifier())
    ])
    
    return pipeline

model=full_pipeline()
# model.fit(x_train, y_train)
# model_pred_test=model.predict(x_test)
# # print(model_pred_test)
# # print(text_cleaner_wrapper("Please find the attached invoice for your recent cloud subscription."))
# print(confusion_matrix(y_test, model_pred_test))
# print(classification_report(y_test, model_pred_test))

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

# 1. Create the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words=None, 
        ngram_range=(1,2),
        max_features=5000,

    )),
    ('rf', RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        max_depth=None,
        class_weight='balanced_subsample'
    ))
])

pipeline.fit(x_train, y_train)
model_pred_test=pipeline.predict(x_test)
print(confusion_matrix(y_test, model_pred_test))
print(classification_report(y_test, model_pred_test))
# 2. Define the parameter grid
# Note: Use 'stepname__parametername' syntax
param_grid = {
    # Tfidf: Keep it tight to avoid the 'curse of dimensionality'
    'tfidf__max_features': [2000, 5000], 
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__stop_words': ['english', None],

    # RandomForest: Focus on controlling tree growth
    'rf__n_estimators': [100, 200, 500],
    'rf__max_depth': [None, 20, 50],             # Limited depth prevents memorizing noise
    'rf__min_samples_leaf': [1, 2, 4],           # Higher values reduce overfitting
    'rf__class_weight': ['balanced', 'balanced_subsample', None] # Crucial for that 74:26 split
}

# 3. Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=20,  # Number of parameter settings sampled
    cv=5,        # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,   # Use all available cores
    verbose=2,
    random_state=42
)

# 4. Fit to your data (assuming X_train and y_train are ready)
# random_search.fit(x_train, y_train)
# print(random_search.best_params_)


# A few test cases
# test_emails = [
#     "Hey, are we still meeting for the AIML group study at 5?", # Should be Ham
#     "CONGRATULATIONS! You've won a $1000 Walmart gift card. Click here to claim now!", # Should be Spam
#     "Please find the attached invoice for your recent cloud subscription." # Should be Ham
# ]

# # The pipeline handles the raw text automatically!
# predictions = full_pipeline.predict([test_emails])
# probabilities = full_pipeline.predict_proba([test_emails])

# for email, pred, prob in zip(test_emails, predictions, probabilities):
#     label = "SPAM" if pred == "spam" else "HAM"
#     conf = max(prob) * 100
#     print(f"Result: {label} ({conf:.2f}% confidence) | Text: {email[:50]}...")
filename = 'spam_model_rf.joblib'
joblib.dump(pipeline, filename)