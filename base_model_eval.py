import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def base_model_eval(X_white, y_white):

    overall_accuracies = {}
    f1_macro_averages = {}


    ml_models = {'logistic regression': LogisticRegression, 
                  'k neighbours': KNeighborsClassifier, 
                  'svm': SVC,
                  'decision tree': DecisionTreeClassifier,
                  'random forest': RandomForestClassifier}


    for name, model in ml_models.items(): 


        # Train | Test split
        X_train, X_test, y_train, y_test = train_test_split(X_white, y_white, test_size=0.3, random_state=101)
        
        # Scale the data
        scaler = StandardScaler()
        scaler.fit(X_train)

        scaled_X_train = scaler.transform(X_train)
        scaled_X_test = scaler.transform(X_test)

        base_model = model()
        base_model.fit(scaled_X_train, y_train)

        y_pred = base_model.predict(scaled_X_test)

        acc = round (accuracy_score(y_test,y_pred), 2)
        f1_macro_avg = round (f1_score(y_test,y_pred, average = 'macro'), 2)

        overall_accuracies[f'{name}'] = acc
        f1_macro_averages[f'{name}'] = f1_macro_avg

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.title(f'{name}')
        plot_confusion_matrix(base_model,X_test,y_test, ax = ax)
        plt.xticks (rotation = 90)
        plt.show()

    print (f'overall accuracy: {overall_accuracies}')
    print (f'f1_macro_averages: {f1_macro_averages}')


"""
def perform_CV(X_white, y_white):
    overall_accuracies = {}
    f1_macro_averages = {}


    ml_models = {'logistic regression': LogisticRegression, 
                  'k neighbours': KNeighborsClassifier, 
                  'svm': SVC,
                  'decision tree': DecisionTreeClassifier,
                  'random forest': RandomForestClassifier}

    
    C = np.logspace(0.001, 100, 10)
    penalty = 
    param_grid = {}


    C = np.linspace(0.001,50,10)

grid_model = GridSearchCV(log_model,param_grid={'C':C,'penalty':penalty}, scoring = 'f1_micro')
"""
