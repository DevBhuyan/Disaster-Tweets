#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 01:26:06 2024

@author: dev
"""
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import spacy
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


nlp = spacy.load("en_core_web_sm")


def preprocess_line(text: str):
    if not isinstance(text, str):
        text = text.str

    doc = nlp(text.lower())

    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]

    return " ".join(lemmatized_tokens)


def fill_missing(series):
    return series.fillna("missing")


def extract_features(df):

    df['lemmatized_tokens'] = df['text'].apply(preprocess_line)

    df['keyword'] = fill_missing(df['keyword'])
    df['location'] = fill_missing(df['location'])

    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['char_count'] = df['text'].apply(lambda x: len(x))
    df['hashtag_count'] = df['text'].apply(lambda x: x.count('#'))
    df['mention_count'] = df['text'].apply(lambda x: x.count('@'))

    return df


def load_train_data():
    if os.path.exists('./processed_train.pkl'):
        with open('./processed_train.pkl', 'rb') as f:
            train_df = pickle.load(f)

    else:
        train_df = pd.read_csv('./train.csv')

        train_df = extract_features(train_df)

        with open('./processed_train.pkl', 'wb') as f:
            pickle.dump(train_df, f)

    return train_df


def load_test_data():
    if os.path.exists('./processed_test.pkl'):
        with open('./processed_test.pkl', 'rb') as f:
            test_df = pickle.load(f)

    else:
        test_df = pd.read_csv('./test.csv')

        test_df = extract_features(test_df)

        with open('./processed_test.pkl', 'wb') as f:
            pickle.dump(test_df, f)

    return test_df


def load_vectorizer():
    text_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
    keyword_vectorizer = TfidfVectorizer()
    location_vectorizer = TfidfVectorizer()
    hashtag_count_scaler = StandardScaler()
    mention_count_scaler = StandardScaler()

    vectorizer = ColumnTransformer(
        transformers=[
            ('text', text_vectorizer, 'lemmatized_tokens'),
            ('keyword', keyword_vectorizer, 'keyword'),
            ('location', location_vectorizer, 'location'),
            ('hashtag_count', hashtag_count_scaler, 'hashtag_count'),
            ('mention_count', mention_count_scaler, 'mention_count')
        ]
    )

    return vectorizer


def load_preprocessed_data():
    train_df = load_train_data()
    test_df = load_test_data()
    vectorizer = load_vectorizer()

    X_train_full = vectorizer.fit_transform(train_df)
    y_train_full = train_df['target']

    X_test = vectorizer.transform(test_df)

    return X_train_full, y_train_full, X_test


def train(X_train,
          y_train,
          model: MultinomialNB = None):

    if not model:
        model = MultinomialNB()

    model.fit(X_train, y_train)

    return model


def train_ensemble(X_train,
                   y_train):

    svc_params = {'C': [1, 10, 100], 'gamma': [1, 0.1, 0.01]}
    rf_param_grid = {
        'n_estimators': [100, 200, 300],   # Number of trees in the forest
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30, None],   # Maximum number of levels in tree
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    clf1 = MultinomialNB(alpha=0.5)
    clf2 = SVC(probability=True)
    clf3 = RandomForestClassifier(n_jobs=-1)

    ensemble_clf = VotingClassifier(
        estimators=[
            ('nb', clf1),
            # ('svc', clf2),
            ('rf', clf3)
        ],
        voting='soft',
        n_jobs=-1)

    ensemble_clf.fit(X_train, y_train)

    return ensemble_clf


def evaluate(y_train, y_train_pred):
    f1 = f1_score(y_train, y_train_pred)
    acc = accuracy_score(y_train, y_train_pred)

    print(round(f1, 4)*100, "%")
    print(round(acc, 4)*100, "%")

    return f1, acc


def submit(model,
           y_test_pred):

    test_df = load_test_data()

    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'target': y_test_pred
    })
    submission_df.to_csv('./submission.csv', index=False)
