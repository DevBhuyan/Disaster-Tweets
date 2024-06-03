#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 03:30:22 2024

@author: dev
"""

from helpers import (
    load_preprocessed_data,
    train_ensemble,
    evaluate,
    submit,
    tqdm
)
from sklearn.model_selection import KFold
import numpy as np

X_train_full, y_train_full, X_test = load_preprocessed_data()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = []
accuracy_scores = []

for train_index, val_index in tqdm(kf.split(X_train_full), total=5):
    X_train, X_val = X_train_full[train_index], X_train_full[val_index]
    y_train, y_val = y_train_full[train_index], y_train_full[val_index]

    model = train_ensemble(X_train, y_train)

    y_val_pred = model.predict(X_val)
    f1, acc = evaluate(y_val, y_val_pred)

    f1_scores.append(f1)
    accuracy_scores.append(acc)

print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")

model = train_ensemble(X_train_full, y_train_full)

y_test_pred = model.predict(X_test)

submit(model, y_test_pred)
