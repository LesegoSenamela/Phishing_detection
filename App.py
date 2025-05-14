# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:49:48 2025

@author: senam
"""

import streamlit as st
import joblib
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import urllib.parse


# Feature Extraction Functions
def extract_features(email_text):
    features = {}

    # Lexical Features
    features['length'] = len(email_text)
    features['num_words'] = len(email_text.split())
    features['num_chars'] = len(email_text.replace(" ", ""))
    features['avg_word_length'] = features['num_chars'] / max(1, features['num_words'])
    features['num_capitals'] = sum(1 for c in email_text if c.isupper())
    features['capitals_ratio'] = features['num_capitals'] / max(1, features['num_chars'])

    # HTML Features
    try:
        soup = BeautifulSoup(email_text, 'html.parser')
        features['has_html'] = 1 if len(soup.find_all()) > 0 else 0
        features['num_links'] = len(soup.find_all('a'))
        features['num_images'] = len(soup.find_all('img'))
    except:
        features['has_html'] = 0
        features['num_links'] = 0
        features['num_images'] = 0

    # URL Features
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text)
    features['num_urls'] = len(urls)
    features['has_url'] = 1 if features['num_urls'] > 0 else 0

    url_features = []
    for url in urls:
        parsed = urllib.parse.urlparse(url)
        url_features.append(len(parsed.path))
        url_features.append(len(parsed.query))
        url_features.append(1 if '@' in parsed.netloc else 0)
        url_features.append(parsed.netloc.count('.'))

    features['avg_url_path_length'] = np.mean(url_features[::3]) if url_features else 0
    features['avg_url_query_length'] = np.mean(url_features[1::3]) if url_features else 0
    features['url_has_at'] = max(url_features[2::3]) if url_features else 0
    features['avg_url_dots'] = np.mean(url_features[3::3]) if url_features else 0

    return features


def explainer(input_data):
    explainer = LimeTextExplainer(class_names=['Legitimate', 'Phishing'])
    # exp = explainer.explain_instance(input_data, model.predict_proba)
    exp = explainer.explain_instance(input_data, lambda x: model.predict_proba(
        pd.concat([pd.DataFrame([extract_features(t)]).
        join(pd.DataFrame(vectorizer.transform([t]).toarray(), columns=[f'tfidf_{i}' for i in range(vectorizer.transform([t]).shape[1])]))
        for t in x])))

    st.subheader("Visual Explanation")
    components.html(exp.as_html(), height=700)


def prediction(input_text):
    features_dict = extract_features(user_text)
    features_df = pd.DataFrame([features_dict])

    tfidf_vec = vectorizer.transform([user_text])
    tfidf_df = pd.DataFrame(tfidf_vec.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_vec.shape[1])])

    final_input = pd.concat([features_df, tfidf_df], axis=1)

    return model.predict(final_input)


model = joblib.load('phishingModel.pkl')
vectorizer = joblib.load('tfidf_content.pkl')

st.title('EmailPhish')
st.write("This is an AI-Powered Phishing Email Detection System.")
st.write("The aim of this system is to detect legitimate emails and phishing attacks. You know when you win an iPhone 13 but you didn't enter for any competition or they send you a notification that you have an unclamed package at the Post Office and to pay a certain ammount. \nThis system aims to help you distunguish between the two.\n\n")
# text box
user_text = st.text_area("Please Enter your Email Text here", value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Type Something...", disabled=False, label_visibility="visible")

if st.button("Submit"):
    if user_text.strip() == "":
        st.warning("Please enter some text before submitting.")
    else:
        st.success("Text submitted successfully!")

        result = prediction(user_text)
        explainer(user_text)

# upload a file
user_file = st.file_uploader("Please upload you Email .txt file here", type="txt", key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

if user_file is None:
    st.warning("Please upload a file to analyse.")

else:
    file_data = user_file.read().decode("utf-8")
    st.success("File submitted successfully!")

    result = prediction(file_data)
    explainer(file_data)
