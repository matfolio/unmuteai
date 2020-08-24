#!/usr/bin/env python
# coding: utf-8

# # UnmuteAI Platform
# 
# The aim of the platform is to develop an application that can be used to predict the sentiment of a perceived review using a content-based filtering algorithm and sentiment analysis combined. The resulting engine is a sentiment-based recommender engine that outputs list of recommended games having closer sentiment values with the prediction probability of the trained classifiers. Unmuteai will take reviews and classify it as either positive sentiment or negative sentiment and also return list of recommended games. 
# 
# The fields of Natural Language Processing and Machine learning (including Deep learning) scope are put to test for the development of this unmuteai. The dataset used are scraped contents from Metascritc website containing game reviews and other game features from critics. Adequate request has been made towards acquiring permissions for retrieving the content from the website using both Request and BeautifulSoup API (Python libraries). The dataset contains 8671 rows and 8 columns. Some other columns are featured extracted using already present once. 
# 
# The Dataset has no missing values, the reason being that some checking procedures where implemented when web scraping the data. Necessary Data cleaning using modules and utilities provided by the NLTK and TextBlob Libraries are performed on the the dataset. The normalisation with regards to removal of HTML tags have been done in the web scraping process. About 5 models are fitted on the cleaned and preporcessed dataset. All models outperformed the benchmarked accuracy score except the the model created using custom word embeddings.  

# In[ ]:


get_ipython().system('pip install nltk')


# In[ ]:


nltk.download()


# In[2]:


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


# In[3]:


nltk.download('stopwords')


# ## Loading Game Review dataset from the system 

# ## Data Loading and Initial exploration 

# In[5]:


print(os.getcwd())
game_reviews = os.getcwd() + "\data\game_reviews.csv"
print(game_reviews)

# Loading Data
df = pd.read_csv(game_reviews)

# Performing initial exploration
# Data shape
print(df.shape)

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


# ## Data Preparation and Cleaning
# 
# - *Removing numeric values*
# - *Removing punctuations*
# - *performing contractions (dealing text inflections)*
# - *Normalizing the text: includes transforming text into lowercase, removing tags if found, filter only alphanumeric features*
# - *Transforming words into their base shape*
# - *Removing stop words*
# 
# The following approach has been taken to categorize the metascore scheme definitions and categorization given by metacritic using their weighted average algorithm.
# 
# **<center>In the dataset, this is labled as pos: target - pos and UI: assigned a green color</center>**
# 
# | indications | metascore | label |
# | :-: | :-: | :-: |
# | Universal acclaim    | 90–100 | pos |
# | Generally favorable reviews with some mixed positive oriented reviews    | 66–89 | pos |
# 
#        
# 
# **<center>In the dataset, this is labled as pos: target - neg and UI: assigned a red color</center>**
# 
# | indications | metascore | label
# | :-: | :-: | :-: |
# | Generally unfavorable reviews with some mixed negative oriented reviews   | 20–65 | neg |
# | Overwhelming dislike    | 0–19 | neg |
#            

# In[6]:


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


# In[7]:


# Feature engineering: creating a new column off the REVIEW column 

df['PROCESSED_REVIEW'] = df['REVIEW'].apply(lambda row: " ".join(get_cleaned_review(row)))

df['RELEASE_YEAR'] = df['RELEASE_DATE'].apply(get_release_year)

df['TOKENIZED_REVIEW'] = df['PROCESSED_REVIEW'].apply(lambda row: [token for token in word_tokenize(row)])

df["LABEL"] = df.METASCORE.apply(lambda metascore: metascore_to_label(metascore))

df['REVIEW_WORD_COUNT'] = df['TOKENIZED_REVIEW'].apply(len)

df['SENTIMENT'] = df['PROCESSED_REVIEW'].apply(lambda review: TextBlob(review).sentiment.polarity)

df['SUBJECTIVITY'] = df['PROCESSED_REVIEW'].apply(lambda review: TextBlob(review).sentiment.subjectivity)


# In[9]:


# Processed dataset
df.columns


# In[10]:


df[:10]


# In[24]:


df.groupby("LABEL").describe()


# In[11]:


## saving the processed dataframe as a binary file using pickle
# The content could be retrieved later from backup using pd.read_pickle(binary)
#df.to_pickle("processed_data.pickle")
df.to_csv(".\data\processed_data.csv")


# ## Exploratory Data Analysis (EDA)

# In[17]:


from nltk import FreqDist
import seaborn as sns
sns.set(font_scale=1.5)
sns.set(rc={'figure.figsize':(20.7,10.27)})
freq_dist_pos = FreqDist(X for i in df.TOKENIZED_REVIEW for X in i if len(X) > 3) 
most_common = freq_dist_pos.most_common(10)
most_common_df_ = pd.DataFrame(most_common, columns = ["WORD","COUNT"])
plot = sns.barplot(x='WORD',y="COUNT", data=most_common_df_)
plot.set(xlabel="most common words", ylabel = "# occurences", title = "Plot of the most common words...")


# In[13]:


# Most common words
most_common_df_


# In[14]:


df['DEVELOPER'].value_counts().head(30).plot(kind='bar', figsize=(12,10))


# In[27]:


# Plot showing the game releases by year.
sns.set(font_scale=1.5)
sns.set(rc={'figure.figsize':(20.7,10.27),"axes.titlesize":19,"axes.labelsize":19})
plot = sns.countplot(x='RELEASE_YEAR', data=df)
plot.set(ylabel="# games released", title = "Plot of game releases by year")


# In[65]:


#df.filter(["REVIEW", "METASCORE", "RELEASE_YEAR"])
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 22}

plt.rc('font', **font)
df.groupby('LABEL').REVIEW.count().plot.bar(figsize=(15, 8))
plt.title("LABEL distribution")
plt.ylabel('Occurrences', fontsize=15)
plt.xlabel('LABEL', fontsize=15)
plt.legend()
plt.show()


# In[66]:


# Plot showing the most used platforms
sns.set(font_scale=1.5)
sns.set(rc={'figure.figsize':(20.7,10.27),"axes.titlesize":19,"axes.labelsize":19})
plot = sns.countplot(x='PLATFORM', data=df)
plot.set(ylabel="# games", title = "Plot of game compatibility with platforms.")


# ## Baseline | Benchmark Accuracy

# In[34]:


## displaying the number of observations with negative sentiment in the dataset.
print("Negative Label: {}".format((df["LABEL"] == 0).sum()))

## displaying the number of observations with positive sentiment in the dataset.
print("Positive Label: {}".format((df["LABEL"] == 1).sum()))

## The baseline accuracy
## this the ratio of the majority class (pos label) to the size of the dataset.
## This would be used as a benchmark when evaluating the minimum accuracy for the models
print("Benchmark Accuracy: {:.2f}".format((df["LABEL"] == 1).sum()/len(df)))


# ## Training and Testing Set

# In[35]:


df = utils.shuffle(df)
# isolating the target column (label)
y = df['LABEL']

X = df.drop(['LABEL'], axis=1)

# Splitting into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Displaying the shape of the features and target section of the dataset
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[38]:


# proportion 0 and 1 in both train and test set
print("***** feature split *****")
print((y_test == 0).sum())
print((y_test == 1).sum())
print('\n')
print("***** target split ******")
print((y_train == 0).sum())
print((y_train == 1).sum())


# ## Developing models and Fitting them on the Extracted features

# #### 1. Fitting the Naive Bayes model on the training set and perform performance evaluation on it 

# In[74]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,roc_curve,roc_auc_score,f1_score,auc,precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer

pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('nb', MultinomialNB())])
parameters_ = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'nb__alpha': [1, 1e-1, 1e-2]
}

#cls_naivebayes_ = MultinomialNB()
nb_clf = GridSearchCV(pipeline, param_grid=parameters_, cv=5)
nb_clf.fit(X_train["PROCESSED_REVIEW"].values, y_train)

print(nb_clf.best_params_);print('\n')
y_pred = nb_clf.predict(X_test["PROCESSED_REVIEW"].values)
# predict probabilities
clf_probs = nb_clf.predict_proba(X_test["PROCESSED_REVIEW"].values)

## Perforamnce metrics
print(classification_report(y_test, y_pred, digits=3));print('\n')
print('Accuracy Score: {:.2f}\n\n'.format(accuracy_score(y_test, y_pred)))
print('Confusion matrix Score: {}\n'.format(confusion_matrix(y_test, y_pred, labels=[1,0])))

# Filtering out the Negative actual ovservations
ns_probs = [0 for _ in range(len(y_test))]
# Positive prediction probabilities
# first index being the negative sentiment...
clf_probs = clf_probs[:, 1]
## calculating at each threshold both precision and recall
clf_precision, clf_recall, _ = precision_recall_curve(y_test, y_pred)

# calculate f1_score
clf_f1 = f1_score(y_test, y_pred)
# calculate auc score
clf_auc = auc(clf_recall, clf_precision)

print("f1 Score {:.2f}\n".format(clf_f1))
print("AUC Score {:.2f}\n".format(clf_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test, clf_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive_Bayes')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
# show the legend
plt.legend()
# show the plot
plt.show()
print(y_pred[300:310])
print(y_test[300:310].values)


# In[151]:


## Testing how good the model is...
X_pred = clf.predict(["best games"])
## Prediction
print(X_pred)

X_pred = clf.predict(["It is a lovely game but I had a really bad experience while playing it..."])
## Prediction
print(X_pred)

X_pred = clf.predict_proba(["best games"])
## Prediction
print(X_pred)

X_pred = clf.predict_proba(["It is a lovely game but I had a really bad experience while playing it..."])
## Prediction
print(X_pred)


# ### 2. Fitting RandomForestClassifier on the train set

# In[75]:


from sklearn.ensemble import RandomForestClassifier

#rf_classifier_ = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 100)
 #'clf__n_estimators': [20],
#'clf__max_features': ['auto', 'sqrt', 'log2'],
#'clf__max_depth' : [4,5,6,7,8]

pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(n_estimators = 20,random_state = 200))])
parameters_ = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__criterion' :['gini', 'entropy']
}


rf_clf = GridSearchCV(pipeline, param_grid=parameters_, cv=5)
rf_clf.fit(X_train["PROCESSED_REVIEW"].values, y_train)

print(rf_clf.best_params_);print('\n')

y_pred = rf_clf.predict(X_test["PROCESSED_REVIEW"].values)
# predict probabilities
clf_probs = rf_clf.predict_proba(X_test["PROCESSED_REVIEW"].values)

## Perforamnce metrics
print(classification_report(y_test, y_pred, digits=3));print('\n')
print('Accuracy Score: {:.2f}\n\n'.format(accuracy_score(y_test, y_pred)))
print('Confusion matrix Score: {}\n'.format(confusion_matrix(y_test, y_pred, labels=[1,0])))

# Filtering out the Negative actual ovservations
ns_probs = [0 for _ in range(len(y_test))]
# Positive prediction probabilities
# first index being the negative sentiment...
clf_probs = clf_probs[:, 1]
## calculating at each threshold both precision and recall
clf_precision, clf_recall, _ = precision_recall_curve(y_test, y_pred)

# calculate f1_score
clf_f1 = f1_score(y_test, y_pred)
# calculate auc score
clf_auc = auc(clf_recall, clf_precision)

print("f1 Score {:.2f}\n".format(clf_f1))
print("AUC Score {:.2f}\n".format(clf_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test, clf_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Model Skill')
plt.plot(nb_fpr, nb_tpr, marker='.', label='RandomForest')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
# show the legend
plt.legend()
# show the plot
plt.show()

print(y_pred[300:310])
print(y_test[300:310].values)


# ### 3. Fitting SVM method (SVC) on the train set 

# In[88]:


from sklearn import svm

parameters_ = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear']
}

pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('svc', svm.SVC(probability=True))])

svm_clf = GridSearchCV(pipeline, param_grid=parameters_, cv=5)
svm_clf.fit(X_train["PROCESSED_REVIEW"].values, y_train)
print(svm_clf.best_params_);print('\n')

y_pred = svm_clf.predict(X_test["PROCESSED_REVIEW"].values)

# predict probabilities
clf_probs = svm_clf.predict_proba(X_test["PROCESSED_REVIEW"].values)

## Perforamnce metrics
print(classification_report(y_test, y_pred, digits=3));print('\n')
print('Accuracy Score: {:.2f}\n\n'.format(accuracy_score(y_test, y_pred)))
print('Confusion matrix Score: {}\n'.format(confusion_matrix(y_test, y_pred, labels=[1,0])))

# Filtering out the Negative actual ovservations
ns_probs = [0 for _ in range(len(y_test))]
# Positive prediction probabilities
# first index being the negative sentiment...
clf_probs = clf_probs[:, 1]
## calculating at each threshold both precision and recall
clf_precision, clf_recall, _ = precision_recall_curve(y_test, y_pred)

# calculate f1_score
clf_f1 = f1_score(y_test, y_pred)
# calculate auc score
clf_auc = auc(clf_recall, clf_precision)

print("f1 Score {:.2f}\n".format(clf_f1))
print("AUC Score {:.2f}\n".format(clf_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test, clf_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(nb_fpr, nb_tpr, marker='.', label='SVM')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
# show the legend
plt.legend()
# show the plot
plt.show()

print(y_pred[300:310])
print(y_test[300:310].values)


# ### 4. Logistic regression

# In[86]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, roc_auc_score

parameters_ = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'lr__C': [0.1, 1, 10]
}

pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('lr',  LogisticRegression())])

LR_clf = GridSearchCV(pipeline, param_grid=parameters_, cv=5)
LR_clf.fit(X_train["PROCESSED_REVIEW"].values, y_train)

print(LR_clf.best_params_);print('\n')
# Predict with the unseen set (testing set)
y_pred = LR_clf.predict(X_test["PROCESSED_REVIEW"].values)

# predict probabilities
clf_probs = LR_clf.predict_proba(X_test["PROCESSED_REVIEW"].values)

## Perforamnce metrics
print(classification_report(y_test, y_pred, digits=3));print('\n')
print('Accuracy Score: {:.2f}\n\n'.format(accuracy_score(y_test, y_pred)))
print('Confusion matrix Score: {}\n'.format(confusion_matrix(y_test, y_pred, labels=[1,0])))

# Filtering out the Negative actual ovservations
ns_probs = [0 for _ in range(len(y_test))]
# Positive prediction probabilities
# first index being the negative sentiment...
clf_probs = clf_probs[:, 1]
## calculating at each threshold both precision and recall
clf_precision, clf_recall, _ = precision_recall_curve(y_test, y_pred)

# calculate f1_score
clf_f1 = f1_score(y_test, y_pred)
# calculate auc score
clf_auc = auc(clf_recall, clf_precision)

print("f1 Score {:.2f}\n".format(clf_f1))
print("AUC Score {:.2f}\n".format(clf_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test, clf_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(nb_fpr, nb_tpr, marker='.', label='Logistic Regression')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
# show the legend
plt.legend()
# show the plot
plt.show()

print(y_pred[300:310])
print(y_test[300:310].values)


# ### 5. Basic Neural Network Using Keras: Training word Embedding and fit NN on the word vector (word embedding)

# In[ ]:


get_ipython().system('pip install keras')
get_ipython().system('pip install theano')
get_ipython().system('pip install tensorflow')


# In[78]:


from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding


# In[79]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
#encoder = LabelEncoder()
#encoded_dec = encoder.fit_transform(y) 
#one_hot_Y = to_categorical(encoded_dec)

y = df['LABEL'].values
# Encoding reviews: 200 word-dimension
sequence_hash_size = 200
encoded_features = [one_hot(review, sequence_hash_size) for review in df.PROCESSED_REVIEW.values]
#print(encoded_features)
# using 100 words as the max length (Indicating words with relatively close in meaning) for dense word vector
max_length = 100
padded_reviews = pad_sequences(encoded_features, maxlen=max_length, padding='post')

# instantiate the model

model = Sequential()
model.add(Dense(200, activation='relu',input_shape=(max_length,)))
model.add(Dense(200, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(padded_reviews, y,epochs=30, batch_size=32, verbose=1)
print(model.summary())
loss, accuracy = model.evaluate(padded_reviews, y, verbose=0)
print('Accuracy: {:.2f}'.format(accuracy))


# ## Sentiment-Based Recommender Engine

# In[80]:


#vectorizer_model = TfidfVectorizer(max_df=0.9)
#vectorizer_model.fit(X_train["PROCESSED_TEXT"].values)
#  displary vocabulary dictionary
#print(vectorizer_model.vocabulary_)
#print(vectorizer_model.idf_)

vectorizer_model = CountVectorizer()

# encode training set
vect_data = vectorizer_model.fit_transform(df["PROCESSED_REVIEW"].values)
print(list(vectorizer_model.vocabulary_)[:20])

#vect_test_data = vectorizer_model.transform(X_test["PROCESSED_REVIEW"].values)

# encoded vector
print(vect_data.shape)
print(vect_data.toarray())

#print(vect_test_data.shape)
#print(vect_test_data.toarray())

# first 50 features
print(vectorizer_model.get_feature_names()[:50])


# In[168]:


if type(similarity) == np.ndarray:
    print("here...")


# In[81]:


from sklearn.metrics.pairwise import cosine_similarity
import pickle
similarity = cosine_similarity(vect_data,vect_data)
print(similarity[1000:1020])

sim_scores =[]

def sortHelper(item):
    return item[3]

def Sorter(desc,obj):
     obj.sort(reverse=desc,key=sortHelper)

def get_recommended_games(title="",sim=similarity, rec_games=[], review_prob=1):
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
            with open('game_recommendation.data', 'wb') as fd:
                ## Storing binary stream using the file handler...
                pickle.dump(rec_games, fd)
            return rec_games_index,rec_games
        else:
            return "Game not found..."
    elif review_prob >= 0:
        sim_scores = df.loc[df["SENTIMENT"] >= (review_prob-0.5), ["SENTIMENT"]]
        if len(sim_scores['SENTIMENT'].values):
            sim_scores = sim_scores.index[0]
            sim_score = pd.Series(similarity[sim_scores])
            print(list(sim_score.sort_values(ascending= False))[:10])
            rec_games_index = sim_score.sort_values(ascending= False).index[:10]
            rec_games = [df.loc[df.index == game, ["TITLE","PLATFORM","RELEASE_DATE","SENTIMENT"]].values.flatten() for game in rec_games_index]
            Sorter(True,rec_games)
            with open('game_recommendation.data', 'wb') as fd:
                ## Storing binary stream using the file handler...
                pickle.dump(rec_games, fd)
            return rec_games_index,rec_games
        else:
            return "Games not found..."
    else:
        print("Game title not provided...")
get_recommended_games("The Guest",review_prob=1)

