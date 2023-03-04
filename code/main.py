#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water potability

@author: yingying
"""

# %% import libraries
# +++++++++++++++++++++++++++++++++++

from trace import Trace
from readline import get_line_buffer
from curses import color_pair
import matplotlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_optimization import Classifier as OptimizedClassifier
from model_selection import Classifier
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy.random as random
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import preprocessing
import os
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None)
matplotlib.style.use('ggplot')


# %%
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ###############################  READ DATA  ################################
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df = pd.read_csv("water_quality.csv")
df.head()
df.dtypes
df.shape
df.corr()

# %%
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ###########################  DATA PREPROCESSING ############################
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ----------------------- Handling Categorical Variable -----------------------
string_col = df.select_dtypes(include="object").columns
df['ammonia'] = pd.to_numeric(df['ammonia'], errors='coerce')
df['is_safe'] = pd.to_numeric(df['is_safe'], errors='coerce')

df.dtypes

# %%
# --------------------------- Handling Null Values ----------------------------
df.isnull().values.any()
df.isnull().sum()

df[df.isna().any(axis=1)]

# drop rows with Nan values since they are insignificant
df.dropna(inplace=True)
df.isnull().values.any()

# %%
temp = []
string = ''
header = ''
format = ''
for i in range(6):
    for j in range(21):
        if i == 0:
            header += '\\bfseries ' + df.columns[j] + ' & '

            format += 'c '
        if j == 0:
            string = str(round(df.iloc[i, j], 2)) + ' & '

        elif j == 20:
            string = string + str(round(df.iloc[i, j], 2))
            print(string)
            temp.append(string)
            stirng = ''
        else:
            string = string + str(round(df.iloc[i, j], 2)) + ' & '
print(header)
print(temp)

# # %%
# # ---------------------------   DATA EXPLORATION   ----------------------------
sns.set_palette(sns.color_palette("GnBu", 2))


def explore_data(data):
    data.describe().T

    # Compute the correlation matrix
    plt.figure(figsize=(4, 3))
    corr = data.corr()
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0, vmax=1, vmin=-1, cmap="GnBu", linewidths=0.5)
    plt.plot()

    # Ratio of Potability vs Non Potability
    data['is_safe'].value_counts()
    not_safe = data[data['is_safe'] == 0]
    safe = data[data['is_safe'] == 1]

    plt.figure(figsize=(2, 3))
    sns.countplot(x="is_safe", data=data)
    data.loc[:, 'is_safe'].value_counts()
    plt.plot()
    return


explore_data(df)

# %% Plot histogram to see detaisl


def plot_histogram(data):
    plt.figure(figsize=(50, 30))
    for i, col in enumerate(data.columns, 1):
        plt.subplot(5, 5, i)
        sns.histplot(x=data[col], data=data, hue='is_safe',
                     multiple='stack', palette='GnBu', bins=20)
        plt.plot()
    return


plot_histogram(df)

# %% Over Sampling since data is imbalanced and no clear indicators on patterns


X = df.drop('is_safe', axis=1)  # features
y = df['is_safe']  # targets


oversample = SMOTE()
print('Original dataset shape %s' % Counter(y))
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_resampled))


X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
y_resampled = pd.DataFrame(y_resampled)
resampled_df = pd.concat([X_resampled, y_resampled], axis=1)

# %%

explore_data(resampled_df)


# Look for skewness
plot_histogram(resampled_df)


# %% Pairplot
def data_insights(data, x):
    plt.figure(figsize=(60, 60))
    sns.pairplot(data, hue=x,  corner=True)
    plt.title("Looking for Insights in Data")
    plt.legend(x)
    plt.tight_layout()
    plt.plot()
    return
# data_insights(resampled_df, 'is_safe')

# %% Look for Outliers


def plot_boxplot(data):
    plt.figure(figsize=(60, 40))
    for i, col in enumerate(data.columns, 1):
        if col == 'is_safe':
            continue
        plt.subplot(5, 5, i)
        sns.boxplot(x=data.is_safe, y=(data[col]))
        plt.tight_layout()
        plt.plot()
    return


plot_boxplot(resampled_df)

# %%
# ----------------------------- Train-Test-Split ------------------------------


def data_splitting(features, targets, ratio=0.3):
    x = features.values
    y = targets.values

    r = 45
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=ratio, random_state=r)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = data_splitting(X_resampled, y_resampled)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
# ------------------------------ Feature Scaling ------------------------------

sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# %%
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #############################  MODEL SELECTION #############################
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# classifier = Classifier(X_train, X_test, y_train, y_test)
optimizedClassifier = OptimizedClassifier(X_train, X_test, y_train, y_test)

models = [
    ("Logistic Regression", 'LogisticRegression'),
    ("Naive Bayes", 'NaiveBayes'),
    ("Decision Tree", 'DecisionTree'),
    ('Random Forest', 'RandomForest'),
    ('K Neighbour Classifier', 'KNearestNeighbours'),
    ("SVM", 'SVM'),
    ('XGB', 'XGBoost'),
    ('Randomised Search', 'RandomizedSearch'),
    # ('Ada Boost', 'AdaBoost'),
    ('Neural Network', 'MLP')
]


def get_results(models):
    accuracy, precision, recall, f1 = [], [], [], []
    TN, FP, FN, TP = [], [], [], []
    loss = []

    for model_name, func in models:
        classifierFunc = getattr(optimizedClassifier, func)

        (predictions, confusion_matrix,
         classification_report, log_loss) = classifierFunc()
        tn, fp, fn, tp = confusion_matrix.ravel()
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)
        TP.append(tp)

        # Calculate metrics
        accuracy.append(round(accuracy_score(y_test, predictions), 4))
        precision.append(round(precision_score(y_test, predictions), 4))
        recall.append(round(recall_score(y_test, predictions), 4))
        f1.append(round(f1_score(y_test, predictions), 4))
        loss.append(round(log_loss, 4))

    return pd.DataFrame({'Model Name': list(map(lambda model: model[0], models)),
                        'Log Loss': loss,
                         'Accuracy': accuracy,
                         'Precision': precision,
                         'Recall': recall,
                         'F1': f1,
                         'TP': TP,
                         'TN': TN,
                         'FP': FP,
                         'FN': FN, })


result = get_results(models)
print(result)

# %%
fig, ax = plt.subplots()


def view_results(df_model, length):
    df = df_model[["Model Name", "Accuracy", "Precision", "Recall"]]
    df = df.plot(kind='bar')
    plt.legend(['Accuracy', 'Precision', 'Recall'], loc='center left',
               bbox_to_anchor=(1.0, 0.5), title='Evaluation Metrics')


graph = view_results(result, len(models))
