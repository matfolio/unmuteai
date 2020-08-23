from __future__ import print_function
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
#from sklearn.externals import joblib
import joblib
import argparse
import json
import os
    
#pd.core.frame.DataFrame
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir',type=str,default=os.environ['SM_OUTPUT_DATA_DIR'])

    parser.add_argument('--model-dir',type=str,default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--train',type=str,default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    file = os.path.join(args.train,"processed_data.csv")

    # Loading Data
    df = pd.read_csv(file, engine = "python")
    # isolating the target column (label)
    y = df['LABEL']
    X = df.drop(['LABEL'], axis=1)
    
    # Splitting into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    nb_clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('nb', MultinomialNB())])
    parameters_ = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'nb__alpha': [1, 1e-1, 1e-2]
    }
    #cls_naivebayes_ = MultinomialNB()
    clf = GridSearchCV(nb_clf, param_grid=parameters_, cv=5)
    clf.fit(X_train["PROCESSED_REVIEW"].values, y_train)
    
    # saving the model using joblib
    joblib.dump(clf,os.path.join(args.model_dir,'model.joblib'))
        

def input_fn(input_data, content_type='application/json'):
    """Takes request data and de-serializes the data into an object for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:
            - The request Content-Type, for example "application/json"
            - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
        The input_fn is responsible to take the request data and pre-process it before prediction.
    Args:
        input_data (obj): the request data.
        content_type (str): the request Content-Type.
    Returns:
        (obj): data ready for prediction.
    """
    #np_array = encoders.decode(input_data, content_type)
    if content_type == 'application/json': 
        return input_data
    else:
        return input_data
    
def predict_fn(input_data, model):
    """A default predict_fn for Scikit-learn. Calls a model on data deserialized in input_fn.
    Args:
        input_data: input data (Numpy array) for prediction deserialized by input_fn
        model: Scikit-learn model loaded in memory by model_fn
    Returns: a prediction
    """
    print('-----------------------')
    print("input payload {}\n".format(input_data))
    print("------------------------")
        
    pred = model.predict([input_data])
    prob = model.predict_proba([input_data])
    return pred, prob
    

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir,'model.joblib'))
    return clf
