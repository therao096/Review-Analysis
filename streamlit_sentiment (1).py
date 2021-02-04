# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 20:31:25 2020

@author: DELL
"""


import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump, load


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


def predict(message):
 model = load('filename.joblib')
 tfidfconverter = TfidfTransformer()
 loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
 X_TEST = loaded_vec.fit_transform([message]).toarray()
 X_TEST_tfidf = tfidfconverter.fit_transform(X_TEST).toarray()
 predictions = model.predict(X_TEST_tfidf)
 return predictions

st.sidebar.subheader("Review Sentiment Analyzer")
st.title("New Customer Review Sentiment Analyzer")
message = st.text_area("Enter Review", "Type here....")
if st.button("Analyze"):
 with st.spinner("Analyzing the text …"):
     prediction = predict(message)
 st.success("Review Sentiment is :" )
 st.write(prediction)
 st.balloons()
else:
 st.warning("Not sure! Try to add some more words")
 
 