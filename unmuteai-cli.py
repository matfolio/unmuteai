import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import utils
import matplotlib.pyplot as plt
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import contractions as ct
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import Word, TextBlob
import string
import re
import os
import argparse
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


def initial_exploration(arg):
    #print(os.getcwd())
    game_reviews = arg
    print(game_reviews)

    # Loading Data
    df = pd.read_csv(game_reviews)
    # Performing initial exploration
    # Data shape
    print(df.shape)

    # No. of instances (rows) in the dataset...
    print(len(df))

    # Display the top 2 observations
    print(df.head(2))

    # Display the bottom 2 observations
    print(df.tail(2))

    # The column names are prefixed with whitespace, this need to be removed
    df.columns = df.columns.str.strip()

    # Normalizing the column names (change all to uppercase)
    df.columns = df.columns.str.upper()

    # showing the attributes in the dataset
    print(df.columns)

    # Checking for any missing values in the columns
    print(df.isnull().sum())

    # Learning more anout the dataset
    print(df.describe())

    # Estimating the mean metascore 
    mean_score = df["METASCORE"].mean()
    print("The mean score of metascore : {0:.2f}".format(mean_score))

    # Unique games in the dataset
    unique_games = np.unique(df["TITLE"])
    print(len(unique_games))

    # Unique platforms in the dataset
    unique_platform = np.unique(df["PLATFORM"])
    print("The list of platforms contained in the dataset: {}".format(len(unique_platform)))

stemmer = PorterStemmer()
words = stopwords.words("english")
punctuations = string.punctuation

## Processing utilities

# removes inflection: I'll ->I will  He'd -> He had
def remove_contractions(text):
    return ct.fix(text)

# transform words into their base or root form: giving -> give, having -> have, discussion -> discuss, lengthy -> lengthi
def get_stemmed_text(word):
    #word = word.lower()
    #root = stemmer.stem(word)
    #return str(TextBlob(root).correct())
    return stemmer.stem(word)

# hide '-' from the list of puctuations available in the string.puctuation module
def reset_punct():
    holder = string.punctuation
    return re.sub("\''","",holder)

def get_cleaned_review(item):
    #punctuations = reset_punct()
    fixed_text = remove_contractions(item)
    processed_text = [i.lower() for i in re.split(r"\W+", fixed_text)]
    processed_text = [ i for i in processed_text if not i in punctuations]
    processed_text = [ i for i in processed_text if len(i) > 2 ] 
    processed_text = [ i for i in processed_text if i.isalpha()]
    processed_text = [ get_stemmed_text(i) for i in processed_text if not i in words]
    return processed_text

# Deprecated
def metascore_to_label_depr(metascore):
    if metascore > 74:
        return 2
    elif metascore > 49 and metascore < 75:
        return 1
    else:
        return 0

# Label encoding 
def metascore_to_label(metascore):
    if metascore > 65:
        return 1
    else:
        return 0

# Split release date by the delimiter '-': from creating a new column 'RELEASE_YEAR'    
def get_release_year(release_year):
    return release_year.split("-")[2]

def preprocess_data(df):
    df = pd.read_csv(df)
    df['PROCESSED_REVIEW'] = df['REVIEW'].apply(lambda row: " ".join(get_cleaned_review(row)))

    df['RELEASE_YEAR'] = df['RELEASE_DATE'].apply(get_release_year)

    df['TOKENIZED_REVIEW'] = df['PROCESSED_REVIEW'].apply(lambda row: [token for token in word_tokenize(row)])

    df["LABEL"] = df.METASCORE.apply(lambda metascore: metascore_to_label(metascore))

    df['REVIEW_WORD_COUNT'] = df['TOKENIZED_REVIEW'].apply(len)

    df['SENTIMENT'] = df['PROCESSED_REVIEW'].apply(lambda review: TextBlob(review).sentiment.polarity)

    df['SUBJECTIVITY'] = df['PROCESSED_REVIEW'].apply(lambda review: TextBlob(review).sentiment.subjectivity)

    return df.to_csv(".\data\processed_data.csv")

''' def save_processed_data(df):
    df.to_csv(".\data\processed_data.csv") '''

def benchmark_accuracy(df):
    df = pd.read_csv(df)
    ## displaying the number of observations with negative sentiment in the dataset.
    print("Negative Label: {}".format((df["LABEL"] == 0).sum()))

    ## displaying the number of observations with positive sentiment in the dataset.
    print("Positive Label: {}".format((df["LABEL"] == 1).sum()))

    ## The baseline accuracy
    ## this the ratio of the majority class (pos label) to the size of the dataset.
    ## This would be used as a benchmark when evaluating the minimum accuracy for the models
    print("Benchmark Accuracy: {:.2f}".format((df["LABEL"] == 1).sum()/len(df)))

def data_split(df):
    df = pd.read_csv(df)
    df = utils.shuffle(df)
    # isolating the target column (label)
    y = df['LABEL']

    X = df.drop(['LABEL'], axis=1)

    # Splitting into train and test set
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20, random_state=42)

    #train_data = X_train
    #X_train['LABEL'] = y_train
    X_train.to_csv("./data/train.csv")

    #test_data = X_test
    #X_test['LABEL'] = y_test
    X_test.to_csv("./data/test.csv")
    # Displaying the shape of the features and target section of the dataset
    #print(X_train.shape)
    #print(X_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)

def train_nb(train):
    train = pd.read_csv(train)

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
    clf.fit(train["PROCESSED_REVIEW"].values, train['LABEL'])

    filename = './data/models/nb_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

def train_rf(train):
    train = pd.read_csv(train)

    rf_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(n_estimators = 20,random_state = 200))])
    parameters_ = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__criterion' :['gini', 'entropy']
    }


    clf = GridSearchCV(rf_clf, param_grid=parameters_, cv=5)
    clf.fit(train["PROCESSED_REVIEW"].values, train['LABEL'])

    filename = './data/models/rf_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    
def test(test,model):
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    test = pd.read_csv(test)
    clf = pickle.load(open(model, 'rb'))
    y_pred = clf.predict(test["PROCESSED_REVIEW"].values)
    
    print(classification_report(test['LABEL'], y_pred, digits=3));print('\n')
    print('Accuracy Score: {}\n\n'.format(accuracy_score(test['LABEL'], y_pred)))
    print('Confusion matrix Score:\n\n {}\n'.format(confusion_matrix(test['LABEL'], y_pred, labels=[1,0])))
    
    # predictions on 10 observations
    print(y_pred[300:310])
    print(test['LABEL'][300:310].values)



## Prediction of review using the trained model
def predict(review,model):
    clf = pickle.load(open(model, 'rb'))
    ## Testing how good the model is...
    pred = clf.predict([review])
    pred_proba = clf.predict_proba([review])
    ## Prediction
    return pred,pred_proba.flatten()


def sortHelper(item):
    return item[3]

def Sorter(desc,obj):
     obj.sort(reverse=desc,key=sortHelper)

def get_recommended_games(title="",path="",rec_games=[], review_prob=1):
    from sklearn.metrics.pairwise import cosine_similarity
    import pickle

    df = pd.read_csv(path)
    vectorizer_model = CountVectorizer()
    # encode training set
    vect_data = vectorizer_model.fit_transform(df["PROCESSED_REVIEW"].values)
    #print(list(vectorizer_model.vocabulary_)[:20])
    # encoded vector
    #print(vect_data.shape)
    #print(vect_data.toarray())


    # first 50 features
    #print(vectorizer_model.get_feature_names()[:50])
    similarity = cosine_similarity(vect_data,vect_data)
    # similarities among the first 10 games.
    print(similarity[1000:1010])

    #game_title = "*****" "Tony Hawk's Pro Skater 2"
    if title:
        sim_scores = df.loc[df["TITLE"] == title, ["TITLE"]]
        if len(sim_scores['TITLE'].values):
            
            sim_scores = sim_scores.index[0]
            sim_score = pd.Series(similarity[sim_scores])
            print(list(sim_score.sort_values(ascending= False))[:10])
            rec_games_index = sim_score.sort_values(ascending= False).index[:10]
            rec_games = [df.loc[df.index == game, ["TITLE","PLATFORM","RELEASE_DATE","SENTIMENT"]].values.flatten() for game in rec_games_index]
            Sorter(True,rec_games)
            with open('./data/game_recommendation.data', 'wb') as fd:
                ## Storing binary stream using the file handler...
                pickle.dump(rec_games, fd)
            return rec_games_index,pd.DataFrame(rec_games,columns= ["TITLE","PLATFORM","RELEASE_DATE","SENTIMENT"])
        else:
            return "Game not found..."
    elif review_prob >= -1:
        sim_scores = df.loc[df["SENTIMENT"] <= review_prob, ["SENTIMENT"]]
        if len(sim_scores['SENTIMENT'].values):
            sim_scores = sim_scores.index[0]
            sim_score = pd.Series(similarity[sim_scores])
            #print(list(sim_score.sort_values(ascending= False))[:10])
            rec_games_index = sim_score.sort_values(ascending= False).index[:10]
            rec_games = [df.loc[df.index == game, ["TITLE","PLATFORM","RELEASE_DATE","SENTIMENT"]].values.flatten() for game in rec_games_index]
            Sorter(True,rec_games)
            with open('./data/game_recommendation.data', 'wb') as fd:
                ## Storing binary stream using the file handler...
                pickle.dump(rec_games, fd)
            return rec_games_index,pd.DataFrame(rec_games,columns= ["TITLE","PLATFORM","RELEASE_DATE","SENTIMENT"])
        else:
            return "Games not found..."
    else:
        print("Game title not provided...")


#print(get_recommended_games("The Guest",path="./data/processed_data.csv",review_prob=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="The tags provided are used to interact with the application to perform\
        various operations on the dataset and apply the models on the processed dataset to learn and get the parameters\
        values estimated")

    parser.add_argument('-t','--train', help="can be used to train the models. used together with\
        with the model tag. ",default="./data/train.csv")
    parser.add_argument('-d','--test', help="can be used to test the models. used together with\
        with the model tag.", default="./data/test.csv")
    parser.add_argument('-p','--predict',help="can be used to predict with the models. used together with\
        with the model tag. Predictions are also displayed together with the game recommendations", default="")
    #parser.add_argument('--data', help="" , default="./data/game_reviews.csv")
    parser.add_argument('-i','--init' ,help="Providing data for initialization", default="./data/game_reviews.csv")
    parser.add_argument('-s','--split' ,help="splitting the dataset into train ans test set.\
         This is required before training models or stretch\
         for prediction of reviews",default="./data/processed_data.csv")
    parser.add_argument('-o','--preprocess', help="Data preprocessing. Data needed to be preprocessed before performing\
         executing the split-data command" , default="./data/game_reviews.csv")
    parser.add_argument('-b','--benchmark', help="Estimate the baseline accuracy. This is needed for performance evaluation\
         of the model. This should be compared with the accuracy result generated for each model." , default="./data/processed_data.csv")
    parser.add_argument('-r','--run' ,help="run executed for every commands sent to the module", type=str, default='')
    parser.add_argument('-m','--model',help="used alongside with models. Provided models: naive_bayes and rf (random forest).\
         These are both used for binary classification." , type=str, default='naive_bayes')
    parser.add_argument('-l','--title' ,help="Needed for recommendations by title. used together with predict tag.", type=str, default='')
    args = parser.parse_args()
    print(args)
    result = ""
    try:
        if args.init and args.run == 'init':
            initial_exploration(args.init)

        if args.preprocess and args.run == 'preprocess':
            preprocess_data(args.preprocess)

        if args.run == 'benchmark':
            benchmark_accuracy(args.benchmark)

        if args.split and args.run == 'split-data':
            data_split(args.split)

        if args.train and args.run == "train":
            if args.model == "naive_bayes":
                train_nb(args.train)
            if args.model == "rf":
                train_rf(args.train)

        if args.run == 'test':
            if args.model == "naive_bayes":
                test(args.test,'./data/models/nb_model.sav')
            if args.model == "rf":
                test(args.test,'./data/models/rf_model.sav')

        if args.predict and args.run == 'predict' :
            print("\n******\n")
            try:
                if args.model == "naive_bayes":
                    print('model in use:\n {}\n'.format(args.model))
                    print("**** Prediction *****")
                    prediction, pred_proba = predict(args.predict,'./data/models/nb_model.sav')
                    print(prediction)
                    try:
                        if prediction[0] == 0:
                            review_prob = pred_proba[0] - 1
                            print(get_recommended_games(args.title,path="./data/processed_data.csv",review_prob=review_prob))
                        else:
                            review_prob = pred_proba[1]
                            print(review_prob)
                            print(get_recommended_games(args.title,path="./data/processed_data.csv",review_prob=review_prob))
                    except:
                        print("The error might be caused by few things: check the path if it exist...")
            except:
                print("model not specified.")
            try:
                if args.model == "rf":
                    print('model in use:\n {}\n'.format(args.model))
                    print("**** Prediction *****")
                    print(predict(args.predict,'./data/models/rf_model.sav'))
                    #get_recommended_games("The Guest",df=args.preprocess,review_prob=0)
            except:
                print("There is an error in the argument...")

            print("\n******\n")
    except:
        print("There seems to be an issue with the Arguments...")

            
        




