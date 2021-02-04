#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

import os
os.getcwd()
os.chdir("F:\\EXCEL R\\project\\final")

# In[2]:


#df = pd.read_csv("C:\\Users\\DELL\\allreviews_samsung.csv")
df = pd.read_csv("C:\\Users\\DELL\\P30_Group5_ExcelR_Project\\Samsung_SDcardReviews_latest.csv")
df = df.drop_duplicates()


df['rating'] = df['stars'].apply(lambda x: re.search(r'\d+',x).group(0))
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.rename(columns ={"comment": "text"})

#df = df.rename(columns ={"comment": "text", "stars": "rating"})
df.info()


# df['token_length'] = [len(x.split(" ")) for x in df.text]
# max(df.token_length)

# df.loc[df.token_length.idxmax(),'text']

# In[3]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" } 


# In[4]:


import codecs
import unidecode
import re
import spacy
nlp = spacy.load("en_core_web_sm")

def spacy_cleaner(text):
    try:
        decoded = unidecode.unidecode(codecs.decode(text, 'unicode_escape'))
    except:
        decoded = unidecode.unidecode(text)
    apostrophe_handled = re.sub("’", "'", decoded)
    expanded = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in apostrophe_handled.split(" ")])
    parsed = nlp(expanded)
    final_tokens = []
    for t in parsed:
        if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@'):
            pass
        else:
            if t.lemma_ == '-PRON-':
                final_tokens.append(str(t))
            else:
                sc_removed = re.sub("[^a-zA-Z]", '', str(t.lemma_))
                if len(sc_removed) > 1:
                    final_tokens.append(sc_removed)
    joined = ' '.join(final_tokens)
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
    return spell_corrected


# In[ ]:


df['clean_text'] = [spacy_cleaner(t) for t in df.text]


# In[ ]:


#define sentiment class
# decide sentiment as positive, negative and neutral 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    if(score['compound']>=0.1):
        return 'pos'
    elif(score['compound']<=-0.1):
        return 'neg'
    else:
        return 'neu'
  

df['sentiment'] = df['clean_text'].apply(lambda x: sentiment_analyzer_scores(x))   
df.sentiment.value_counts()


# In[ ]:


def sentiment_analyzer_score_values(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    return score['compound']
    
df['sentiment_score'] = df['clean_text'].apply(lambda x: sentiment_analyzer_score_values(x))   
#df.sentiment_score.head(5)


# In[ ]:



from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=5000, min_df=5, max_df=0.7, stop_words=None, decode_error="replace")
X = vectorizer.fit_transform(df.clean_text).toarray()
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X_tfidf = tfidfconverter.fit_transform(X).toarray()
#Save vectorizer.vocabulary_
import pickle
pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df.sentiment, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


################                   dealing with imbalanced data             ###################
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train,np.array(y_train).ravel())
print(X_train_res.shape, y_train.shape)

# Classifier - Algorithm - SVM
from sklearn import svm
from sklearn.svm import SVC
# fit the training dataset on the classifier
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
################                   dealing with imbalanced data             ###################
###  building svm model after applying SMOTE 
smote_svm = SVM.fit(X_train_res,y_train_res).predict(X_test)
# summarize the fit of the model
from sklearn import metrics
print(metrics.classification_report(y_test, smote_svm))
print(metrics.confusion_matrix(y_test, smote_svm))
print(" accuracy = ", accuracy_score(y_test, smote_svm))
#load the nlp model
pickle.dump(smote_svm,open("nlp_model.pkl","wb"))

# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt 

def generate_wordcloud(data, plot_title):
  wc = WordCloud(width=1299, height=800, max_words=200,random_state=1234, colormap="Dark2").generate(str(data))
  #PLOT
  plt.figure(figsize=(10,8))
  plt.title(plot_title)
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.show()
  
Word_data=df['clean_text'].astype('str').tolist()
generate_wordcloud(Word_data, "word cloud for overall reviews")


# In[ ]:


#sentiment analysis plot
import seaborn as sns
plt.rcParams['figure.figsize'] = (10.0, 6.0)
#plt.rcParams['font.family'] = "serif"
colors = ["red", "green", "yellow"]
p = sns.countplot(data=df, x = 'sentiment', palette = colors)
plt.title('Sentiment Analysis')
plt.show()


# In[ ]:


# bar chart for review ratings
colors = ["red", "orange", "yellow", "blue", "green"]
p = sns.countplot(data=df, x = 'rating', palette = colors)
plt.title('Histogram for Customer Ratings')
plt.figure(figsize=(30,10))
plt.show()

# In[ ]:


plt.title('Percentage of Ratings', fontsize=20)
colors = ["green", "blue", "yellow", "orange", "red"]
df.rating.value_counts().plot(kind='pie', labels=['Rating5', 'Rating4', 'Rating3', 'Rating2', 'Rating1'],
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, colors = colors, 
                              textprops={'fontsize': 15}, )

plt.show()

# In[ ]:


#pie chart for sentiment
plt.title('   Percentage of Reviews', fontsize=20)
colors = ["green", "yellow", "red"]
df.sentiment.value_counts().plot(kind='pie', labels=['Positive Reviews', 'Neutral Reviews', 'Negative Reviews'],
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, colors = colors, 
                              textprops={'fontsize': 15}, )
plt.show()

# In[ ]:


import plotly.express as px
fig =px.sunburst(df, path = ['sentiment', 'rating'],
                 height = 700,
                 template = "ggplot2"
)
fig.show()


# In[ ]:


# applying groupby() function to 
group_bysentiment = df.groupby('sentiment') 
df['sentiment'].value_counts()
pos_review = group_bysentiment.get_group('pos') 
neu_review = group_bysentiment.get_group('neu') 
neg_review = group_bysentiment.get_group('neg') 

pos_data = pos_review['clean_text'].astype('str').tolist()
generate_wordcloud(pos_data, "Word Cloud for positive reviews")

neu_data = neu_review['clean_text'].astype('str').tolist()
generate_wordcloud(neu_data, "Word Cloud for Neutral reviews")

neg_data = neg_review['clean_text'].astype('str').tolist()
generate_wordcloud(neg_data, "Word Cloud for Negative reviews")


# In[ ]:


df.sentiment_score.head(5)


# In[ ]:

plt.xlabel('Rating')
plt.ylabel('Average Sentiment')
plt.title('Average Sentiment per Rating Distribution')
colors=["red", "red", "green", "green", "green"]
polarity_avg = df.groupby('rating')['sentiment_score'].mean().plot(kind='bar', figsize=(10,6), color = colors)
plt.show()


# In[ ]:


# Histogram of polarity score
num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df.sentiment_score, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Number of Reviews')
plt.title('Histogram of Polarity Score')
plt.show()


# In[ ]:


# unigrams
from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['clean_text'], 30)
df2 = pd.DataFrame(common_words, columns = ['unigram' , 'count'])
sns.barplot(data = df2
            ,x = 'count'
            ,y = 'unigram'
            ,color = 'cyan' 
            ,ci = None
            )
plt.title('Top 30 unigrams of reviews')
plt.show()

# In[ ]:


## bigrams
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['clean_text'], 20)
df3 = pd.DataFrame(common_words, columns = ['bigram' , 'count'])

sns.barplot(data = df3
            ,x = 'count'
            ,y = 'bigram'
            ,color = 'cyan' 
            ,ci = None
            )
plt.title('Top 20 bigrams of reviews')
plt.show()

# In[ ]:


## trigrams
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(df['clean_text'], 20)
df4 = pd.DataFrame(common_words, columns = ['trigram' , 'count'])

sns.barplot(data = df4
            ,x = 'count'
            ,y = 'trigram'
            ,color = 'cyan' 
            ,ci = None
            )
plt.title('Top 20 trigrams of reviews')
plt.show()

df = df.to_csv('CleanData_Samsung_SDcardReviews_weekly_update.csv', index = True, encoding = 'utf-8') 
