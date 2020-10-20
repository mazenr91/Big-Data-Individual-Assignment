#!/usr/bin/env python
# coding: utf-8

# # Individual Assignment Question 2 

# In[1]:


# Mazen Al Rifai
# 20198044
# MMA
# 2021W
# MMA 865
# 18 October 2020


# Submission to Question [1], Part [a]


# In[2]:


import pandas as pd
import numpy as np
import re
import string
# For visualizations
import matplotlib.pyplot as plt
from collections import Counter 


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[4]:


import os
os.getcwd()


# # Load Data

# In[5]:


#Import train and test data
df_train = pd.read_csv("sentiment_train.csv")
df_test = pd.read_csv("sentiment_test.csv")
print(df_train.info())
print(df_train.head())
print(df_test.info())
print(df_test.head())


# In[6]:


#Check if data is imbalanced
print('Train:')
df_train['Polarity'].value_counts() 


# In[7]:


#Feature Engineering

#Word Count of each review
df_train['word_count'] = df_train['Sentence'].apply(lambda x: len(str(x).split(" ")))

#Character Count of each review
df_train['char_count'] = df_train['Sentence'].str.len() ## this also includes spaces


#Character count without punctuations
punctuation = ['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~','``',"''",'--']
def count_characters(data_df):
    reviewcharacters = []
    text_col = data_df['Sentence']
    for i in text_col:
        a = dict(Counter(i))
        b = {k:v for k, v in a.items() if k not in punctuation}
        c = sum(list(b.values()))
        reviewcharacters.append(c)
    data_df['reviewChars'] = reviewcharacters
    return data_df['reviewChars']
df_train['reviewChars'] = count_characters(df_train)

#Number of Punctuations

def numpunct(data_df):
    reviewPuncts = []
    for i in data_df['Sentence']:
        a = dict(Counter(i))
        b = {k:v for k,v in a.items() if k in punctuation}
        c = sum(list(b.values()))
        reviewPuncts.append(c)
    data_df['reviewPuncts'] = reviewPuncts
    return data_df['reviewPuncts']
df_train['reviewPuncts'] = numpunct(df_train)

# Ratio of Punctuations to reviewc characters
def ratio_puncts_chars(data_df):
    return data_df['reviewPuncts'] / data_df['reviewChars']
df_train['ratiopunChar'] = ratio_puncts_chars(df_train)

#Number of Capital Words
def numcapwords(data_df):
    reviewCwords = []
    for i in data_df['Sentence']:
        a = i.split()
        b = [word for word in a if word.isupper()]
        c = len(b)
        reviewCwords.append(c)
    data_df['reviewCwords'] = reviewCwords
    return data_df['reviewCwords']
df_train['reviewCwords'] = numcapwords(df_train)


#Sentiment Analysis 
from textblob import TextBlob
df_train['polarity_score']=df_train['Sentence'].apply(lambda x:TextBlob(x).sentiment.polarity)


# In[8]:


#Frequency distribution of Part of Speech Tags
import textblob
pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt
df_train['noun_count'] = df_train['Sentence'].apply(lambda x: check_pos_tag(x, 'noun'))
df_train['verb_count'] = df_train['Sentence'].apply(lambda x: check_pos_tag(x, 'verb'))
df_train['adj_count'] = df_train['Sentence'].apply(lambda x: check_pos_tag(x, 'adj'))
df_train['adv_count'] = df_train['Sentence'].apply(lambda x: check_pos_tag(x, 'adv'))
df_train['pron_count'] = df_train['Sentence'].apply(lambda x: check_pos_tag(x, 'pron'))


# # Custom Functions for Preprocessing and Feature Engineering

# In[9]:


#pip install unidecode


# In[10]:


#Text Preprocessing

# Remove ,,, from entries in name column
df_train['Sentence'] = df_train['Sentence'].str.replace(r'\,,,','')
df_train['Sentence'].unique()

#Remove contractions from text reviews
# Dictionary of English Contractions
contractions_dict = { "It's":"It is","it's":"it is", "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, text)

# Expanding Contractions in the reviews
df_train['Sentence']=df_train['Sentence'].apply(lambda x:expand_contractions(x))

#Lowercase letters
df_train['Sentence']=df_train['Sentence'].str.lower()

#Remove digits and words containing digits 
df_train['Sentence']=df_train['Sentence'].apply(lambda x: re.sub('\w*\d\w*','', x))

#Remove Punctuations
df_train['Sentence']=df_train['Sentence'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

#Commonly occuring words
freq = pd.Series(' '.join(df_train['Sentence']).split()).value_counts()[:10]
freq

#removal of commonly occuring irrelevant words
freq = list(freq)
df_train['Sentence'] = df_train['Sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df_train['Sentence'].head()

#Count of rare words

freq_rare = pd.Series(' '.join(df_train['Sentence']).split()).value_counts()[-100:]
freq_rare

#removal of rarely occuring irrelevant words
freq_rare = list(freq_rare)
df_train['Sentence'] = df_train['Sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df_train['Sentence'].head()

# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
from nltk.corpus import stopwords
stop = stopwords.words('english')
df_train['Sentence.nostopwords'] = df_train['Sentence'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Lemmatization

from textblob import Word
df_train['Sentence.nostopwords.LEMMATIZED'] = df_train['Sentence.nostopwords'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))




# In[11]:


#Generate Word Cloud for train data
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(df_train['Sentence.nostopwords.LEMMATIZED'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# In[12]:


#Plot 10 most common words in train dataset

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(df_train['Sentence.nostopwords.LEMMATIZED'])
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)


# In[13]:


# Submission to Question [1], Part [b]


# In[14]:


#Text to Vector 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=750, lowercase=True, stop_words = 'english', ngram_range=(1, 2))
#fit the vectorizers to the data.

features = vectorizer.fit_transform(df_train['Sentence.nostopwords.LEMMATIZED'])
pandaframe = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names())


# In[15]:


#Topic Modelling
from sklearn.decomposition import LatentDirichletAllocation

lda_model = LatentDirichletAllocation(n_components=10,
                                      doc_topic_prior=None,
                                      topic_word_prior=None,
                                      max_iter=200, 
                                      learning_method='batch', 
                                      random_state=123,
                                      n_jobs=2,
                                      verbose=0)
lda_output = lda_model.fit(features)

# Log Likelyhood: Higher the better
ll = lda_model.score(features)

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
perp = lda_model.perplexity(features)


# In[16]:


# Theta = document-topic matrix
# Beta = components_ = topic-term matrix
theta = pd.DataFrame(lda_model.transform(features))
beta = pd.DataFrame(lda_model.components_)

print('theta:')
theta.head()

#beta
print('beta:')
beta.head(20)



# In[17]:


# Build Topic Summary
feature_names = vectorizer.get_feature_names()
weight = theta.sum(axis=0)
support50 = (theta > 0.5).sum(axis=0)
support10 = (theta > 0.1).sum(axis=0)
termss = list()
for topic_id, topic in enumerate(lda_model.components_):
    terms = " ".join([feature_names[i] for i in topic.argsort()[:-6 - 1:-1]])
    termss.append(terms)
topic_summary = pd.DataFrame({'TopicID': range(0, len(termss)), "Support50": support50, "Support10": support10, "Weight": weight, "Terms": termss})


# In[18]:


#Display topic Summary
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 0)
topic_summary


# In[19]:


# Submission to Question [1], Part [c]


# In[20]:


#Select features to use in model
df1=df_train[['word_count', 'char_count','polarity_score','Polarity','reviewChars','reviewPuncts','ratiopunChar','reviewCwords','noun_count','verb_count','adj_count','adv_count','pron_count']]
df2= pd.concat([pandaframe.reset_index(drop=True), df1.reset_index(drop=True),theta], axis=1)



# In[21]:


#Repace infiniti values with zero, if any to avoid errors when building the model
df2.info()
df2 = df2.replace([np.inf, -np.inf], np.nan)
df2["ratiopunChar"]=df2["ratiopunChar"].fillna(0)


# In[22]:


#The data was split to two parts 80% for training and 20% for testing. 
from sklearn.model_selection import train_test_split

# For X variable, the target variable ("label") was dropped since it will be predicted using the model.
X = df2.drop(["Polarity"], axis=1)

#For y variable, all other variabes are dropped. This variable will be used for assessing the model peformance using the AUC metric
y = df2["Polarity"]
# A random seed was assigned so that our results will be reproducible on the same machine

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df2["Polarity"].values)


# In[23]:


#Logistic Regression classifier with hyperparameter tuning, other models tried: RF and DT
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


parameters_grid = {'penalty':['l1', 'l2'],'dual':[True, False],'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'fit_intercept':[True, False],
        'solver':['saga','lbfgs','liblinear']}

# Create grid search object

LogReg_grid = GridSearchCV(LogisticRegression(random_state=42), parameters_grid, cv=5, n_jobs=2, scoring='f1_weighted')

# Fit on data

get_ipython().run_line_magic('time', 'LogReg_grid.fit(X_train, y_train)')

LogReg_grid.best_params_ # printing out best parameters


# In[24]:


import seaborn as sns 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import accuracy_score

# Predicting performance of hypertuned Logistic Regression model


pred_val = LogReg_grid.predict(X_val)

#Get the confusion matrix
cf_matrix = confusion_matrix(y_val,pred_val)
print(cf_matrix)

tn, fp, fn, tp = confusion_matrix(y_val,pred_val).ravel()
(tn, fp, fn, tp)

sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

print("\nF1 Score = {:.5f}".format(f1_score(y_val, pred_val, average="micro")))
print("AUC Score = {:.3f}".format(roc_auc_score(y_val, pred_val)))
print("Accuracy = {:.3f}".format(accuracy_score(y_val, pred_val)))
print()
print(classification_report(y_val, pred_val))

#1 BOW: F1=0.71202, AUC= 0.711
#2 BOW + Count Features: F1=0.7188, AUC= 0.718
#3 BOW + Count Features + Polarity: F1=0.814, AUC= 0.814


# In[25]:


#Submission to Question [1], Part [d] 


# # Test Data

# In[26]:


#Feature Engineering

#Word Count of each review
df_test['word_count'] = df_test['Sentence'].apply(lambda x: len(str(x).split(" ")))

#Character Count of each review
df_test['char_count'] = df_test['Sentence'].str.len() ## this also includes spaces

#Character count without punctuations
punctuation = ['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~','``',"''",'--']
def count_characters(data_df_test):
    reviewcharacters = []
    text_col = data_df_test['Sentence']
    for i in text_col:
        a = dict(Counter(i))
        b = {k:v for k, v in a.items() if k not in punctuation}
        c = sum(list(b.values()))
        reviewcharacters.append(c)
    data_df_test['reviewChars'] = reviewcharacters
    return data_df_test['reviewChars']
df_test['reviewChars'] = count_characters(df_test)

#Number of Punctuations

def numpunct(data_df_test):
    reviewPuncts = []
    for i in data_df_test['Sentence']:
        a = dict(Counter(i))
        b = {k:v for k,v in a.items() if k in punctuation}
        c = sum(list(b.values()))
        reviewPuncts.append(c)
    data_df_test['reviewPuncts'] = reviewPuncts
    return data_df_test['reviewPuncts']
df_test['reviewPuncts'] = numpunct(df_test)

# Ratio of Punctuations to reviewc characters
def ratio_puncts_chars(data_df_test):
    return data_df_test['reviewPuncts'] / data_df_test['reviewChars']
df_test['ratiopunChar'] = ratio_puncts_chars(df_test)

#Number of Capital Words
def numcapwords(data_df_test):
    reviewCwords = []
    for i in data_df_test['Sentence']:
        a = i.split()
        b = [word for word in a if word.isupper()]
        c = len(b)
        reviewCwords.append(c)
    data_df_test['reviewCwords'] = reviewCwords
    return data_df_test['reviewCwords']
df_test['reviewCwords'] = numcapwords(df_test)


#Sentiment Analysis 
from textblob import TextBlob
df_test['polarity_score']=df_test['Sentence'].apply(lambda x:TextBlob(x).sentiment.polarity)

#Frequency distribution of Part of Speech Tags
df_test['noun_count'] = df_test['Sentence'].apply(lambda x: check_pos_tag(x, 'noun'))
df_test['verb_count'] = df_test['Sentence'].apply(lambda x: check_pos_tag(x, 'verb'))
df_test['adj_count'] = df_test['Sentence'].apply(lambda x: check_pos_tag(x, 'adj'))
df_test['adv_count'] = df_test['Sentence'].apply(lambda x: check_pos_tag(x, 'adv'))
df_test['pron_count'] = df_test['Sentence'].apply(lambda x: check_pos_tag(x, 'pron'))


# In[27]:


#Text Preprocessing

#Word Count of each review
df_test['word_count'] = df_test['Sentence'].apply(lambda x: len(str(x).split(" ")))
df_test[['Sentence','word_count']].head()


#Character Count of each review
df_test['char_count'] = df_test['Sentence'].str.len() ## this also includes spaces
df_test[['Sentence','char_count']].head()

# Remove ,,, from entries in name column
df_test['Sentence'] = df_test['Sentence'].str.replace(r'\,,,','')
df_test['Sentence'].unique()

#Remove contractions from text reviews
# Dictionary of English Contractions
contractions_dict = { "It's":"It is","it's":"it is", "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, text)

# Expanding Contractions in the reviews
df_test['Sentence']=df_test['Sentence'].apply(lambda x:expand_contractions(x))

#Lowercase letters
df_test['Sentence']=df_test['Sentence'].str.lower()

#Remove digits and words containing digits 
df_test['Sentence']=df_test['Sentence'].apply(lambda x: re.sub('\w*\d\w*','', x))

#Remove Punctuations
df_test['Sentence']=df_test['Sentence'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

#Commonly occuring words
freq = pd.Series(' '.join(df_test['Sentence']).split()).value_counts()[:10]
freq

#removal of commonly occuring irrelevant words
freq = list(freq)
df_test['Sentence'] = df_test['Sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df_test['Sentence'].head()

#Count of rare words

freq_rare = pd.Series(' '.join(df_test['Sentence']).split()).value_counts()[-100:]
freq_rare

#removal of rarely occuring irrelevant words
freq_rare = list(freq_rare)
df_test['Sentence'] = df_test['Sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df_test['Sentence'].head()

# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
from nltk.corpus import stopwords
stop = stopwords.words('english')
df_test['Sentence.nostopwords'] = df_test['Sentence'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Lemmatization

from textblob import Word
df_test['Sentence.nostopwords.LEMMATIZED'] = df_test['Sentence.nostopwords'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[28]:


#Text Vectorizer  

#fit the vectorizers to the data

features3= vectorizer.transform(df_test['Sentence.nostopwords.LEMMATIZED'])
pandaframe3 = pd.DataFrame(features3.toarray(), columns=vectorizer.get_feature_names())


# In[29]:


#Topic modelling for test data set
theta_test = pd.DataFrame(lda_model.transform(features3))



# In[30]:


# Build Topic Summary
feature_names = vectorizer.get_feature_names()
weight = theta_test.sum(axis=0)
support50 = (theta_test > 0.5).sum(axis=0)
support10 = (theta_test > 0.1).sum(axis=0)
termss = list()
for topic_id, topic in enumerate(lda_model.components_):
    terms = " ".join([feature_names[i] for i in topic.argsort()[:-6 - 1:-1]])
    termss.append(terms)
topic_summary = pd.DataFrame({'TopicID': range(0, len(termss)), "Support50": support50, "Support10": support10, "Weight": weight, "Terms": termss})


# In[31]:


pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 0)
topic_summary


# In[32]:


#Choose Features and combine word features with other features 
df4=df_test[['word_count', 'char_count','polarity_score','Polarity','reviewChars','reviewPuncts','ratiopunChar','reviewCwords','noun_count','verb_count','adj_count','adv_count','pron_count']]
df5= pd.concat([pandaframe3.reset_index(drop=True), df4.reset_index(drop=True),theta_test], axis=1)


# In[33]:


# Prediction on test dataset
X_test = df5.drop(['Polarity'], axis=1)

# Use model to make predictions
pred_test = LogReg_grid.predict(X_test)


# In[34]:


#Check performance on test dataset

y_val=df5['Polarity']
#Get the confusion matrix
cf_matrix = confusion_matrix(y_val,pred_test)
print(cf_matrix)

tn, fp, fn, tp = confusion_matrix(y_val,pred_test).ravel()
(tn, fp, fn, tp)

sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

print("\nF1 Score = {:.5f}".format(f1_score(y_val, pred_test, average="micro")))
print("AUC Score = {:.3f}".format(roc_auc_score(y_val, pred_test)))
print("Accuracy = {:.3f}".format(accuracy_score(y_val, pred_test)))

print()
print(classification_report(y_val, pred_test))


# In[35]:


# Submission to Question [2]


# In[36]:


#Generate Word Cloud for test data
long_string = ','.join(list(df_test['Sentence.nostopwords.LEMMATIZED'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# In[37]:


#Plot 10 most common words in test dataset
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(df_test['Sentence.nostopwords.LEMMATIZED'])
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)


# In[38]:


#Submission to Question [3] 


# In[39]:


#Generate Comparison table to determine why some predicitons were incorrect 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pred_test_pd = pd.DataFrame({'predicted': pred_test})
df_test_original = pd.read_csv("sentiment_test.csv") # Original Test data without features
df_test_original_Sentence=df_test_original.drop(columns='Polarity')

array=LogReg_grid.predict_proba(X_test)
probs= pd.DataFrame({'Prob 0': array[:, 0], 'Prob 1': array[:, 1]})
                                
Comparison=pd.concat([df_test_original_Sentence.reset_index(drop=True),df_test.reset_index(drop=True),pred_test_pd.reset_index(drop=True),probs.reset_index(drop=True),pandaframe3.reset_index(drop=True),theta_test.reset_index(drop=True)],axis=1)

Comparison.head(30)
Comparison.tail(30)


# In[40]:


Wrong=Comparison[Comparison.Polarity!=Comparison.predicted]
Wrong=Wrong.iloc[:, 0:20]
Wrong

