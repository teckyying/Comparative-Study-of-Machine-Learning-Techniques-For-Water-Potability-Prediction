
"""
@author: yingying
"""

from turtle import pen
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
# print winning set of hyperparameters
from pprint import pprint
from scipy.stats import uniform, truncnorm, randint, loguniform
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.metrics import log_loss

import warnings
warnings.filterwarnings(
    action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')


class Classifier():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # self.feature_col_tree=self.df_tree.columns.to_list()
        # self.feature_col_tree.remove(self.target)

    def results(self, predictions, tree_name):
        print('\n\n--------------------------------------------------- ')
        print("------------------- ", tree_name, "------------------- \n")

        print('Confusion matrix: ')
        cm = confusion_matrix(self.y_test, predictions)
        print(cm, '\n')

        print('Classification Report: ')
        cr = classification_report(self.y_test, predictions)
        print(cr, "\n")

        logLoss = log_loss(self.y_test, predictions)

        return (predictions, cm, cr, logLoss)

    ###########################  Naive Bayes Classifier ########################
    def LogisticRegression(self):
        classifier = LogisticRegression()

        penalty = ['l1', 'l2', 'elasticnet', 'none']
        solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        c = [100, 10, 1.0, 0.1, 0.01]
       

        # Create the param grid
        param_grid = {
            # 'n_estimators': n_estimators,
                    'penalty': penalty,
                      'solver': solver,
                      'C': c}

      
        clf = RandomizedSearchCV(classifier, param_grid, n_iter=100, cv=10)
        model = clf.fit(self.X_train, self.y_train)
        optimized_model = model.best_estimator_
        pprint(optimized_model.get_params())

        #make predictions
        predictions = optimized_model.predict(self.X_test)

        return self.results(predictions, 'Logistic Regression')

   ###########################  Naive Bayes Classifier ########################
    def NaiveBayes(self):
        classifier = GaussianNB()
        
        var_smoothing = np.logspace(0,-9, num=100)
        # var_smoothing = [0.1]
       

        # Create the param grid
        param_grid = {'var_smoothing': var_smoothing}

      
        clf = RandomizedSearchCV(classifier, param_grid, n_iter=500, cv=10)
        model = clf.fit(self.X_train, self.y_train);

        optimized_model = model.best_estimator_
        pprint(optimized_model.get_params())

        #make predictions
        predictions = optimized_model.predict(self.X_test)

        return self.results(predictions, 'Naive Bayes')

    ##############################  Decision Tree #############################
    def DecisionTree(self):
        classifier = DecisionTreeClassifier()
        dt_classifier = classifier.fit(self.X_train, self.y_train)

        # n_estimators = [50, 100, 250, 500, 750, 1000, 1500]
        criterion = ['gini', 'entropy', 'log_loss']
        splitter = ['best', 'random']
        # max_features = ['none', 'log2', 'sqrt']
        min_samples_split =  [2, 5]
        min_samples_leaf = [1, 2, 3]

        # Create the param grid
        param_grid = {
                      'criterion': criterion,
                    'splitter': splitter,
                    #   'max_features': max_features,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
               }


        clf = RandomizedSearchCV(dt_classifier, param_grid, n_iter=100, cv=10)
        model = clf.fit(self.X_train, self.y_train)
  

        # make predictions
        optimized_model = model.best_estimator_
        pprint(optimized_model.get_params())
        predictions = optimized_model.predict(self.X_test)
        # make predictions

        # predictions = decision_tree.predict(self.X_test)

        return self.results(predictions, 'Decision Tree')

    ##############################  Random Forest ##############################

    def RandomForest(self):
        rf_classifier = RandomForestClassifier()

        n_estimators = [50, 100, 250, 500, 750, 1000, 1500]
        criterion = ['gini', 'entropy']
        max_features = ['log2', 'sqrt', 'none']
        min_samples_split = [2, 5]
        min_samples_leaf = [1, 2, 3]
        bootstrap = [True, False]

        # Create the param grid
        param_grid = {'n_estimators': n_estimators,
                      'criterion': criterion,
                      'max_features': max_features,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'bootstrap': bootstrap}


        clf = RandomizedSearchCV(rf_classifier, param_grid, n_iter=50, cv=10)
        model = clf.fit(self.X_train, self.y_train)
  

        # make predictions
        optimized_model = model.best_estimator_
        pprint(optimized_model.get_params())
        predictions = optimized_model.predict(self.X_test)

        return self.results(predictions,  'Random Forest_random')
        # return

    ############################  KNearestNeighbours ############################
    def KNearestNeighbours(self):
        classifier = KNeighborsClassifier()

        # n_neighbors =  [10, 20, 30, 40, 50]
        n_neighbors = [30]
        weights = ['uniform', 'distance']
        metric = ['euclidean', 'manhattan', 'minkowski']
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
      

        # Create the param grid
        param_grid = {
                        'n_neighbors': n_neighbors,
                        'weights': weights,
                        'metric': metric,
                        'algorithm': algorithm,
                     }


        clf = RandomizedSearchCV(classifier, param_grid, n_iter=100, cv=10)
        model = clf.fit(self.X_train, self.y_train)
  

        # make predictions
        optimized_model = model.best_estimator_
        pprint(optimized_model.get_params())
        predictions = optimized_model.predict(self.X_test)


        return self.results(predictions,  'K Nearest Neighbours')

    ##################################  SVM ##################################
    def SVM(self):
        classifier = SVC()

        kernel = ['poly', 'rbf', 'sigmoid',]
        C = [50, 10, 1.0, 0.1, 0.01]
        gamma = ['scale', 'auto']

        # Create the param grid
        param_grid = {
                        'kernel': kernel,
                        'C': C,
                        'gamma': gamma ,
                     }


        clf = RandomizedSearchCV(classifier, param_grid, n_iter=100, cv=10)
        model = clf.fit(self.X_train, self.y_train)
  

        # make predictions
        optimized_model = model.best_estimator_
        pprint(optimized_model.get_params())
        predictions = optimized_model.predict(self.X_test)

        return self.results(predictions,  'SVC')

    #################################  XGBoost ################################
    def XGBoost(self):
        classifier = xgboost.XGBClassifier()

        n_estimators = [50, 100, 250, 500, 750, 1000, 1500]
        booster = ['gbtree', 'gblinear']
        base_score = [0.25, 0.5, 0.75, 1]
        # min_child_weight =[0, 1,2,3,4]
        eta = [0.05, 0.1, 0.15, 0.20]
        eval_metric = ['logloss']

        hyperparameter_grid = {
            'n_estimators': n_estimators,
            'booster': booster,
            'base_score': base_score,
            'eta': eta,
            'eval_metric': eval_metric,
        }
        result = RandomizedSearchCV(estimator=classifier, param_distributions=hyperparameter_grid, n_iter=50, cv=10,
                                    random_state=42, verbose=0)
        result.fit(self.X_train, self.y_train)


        # make predictions
        xgrf2  = result.best_estimator_
        pprint(xgrf2.get_params())


        predictions = xgrf2.predict(self.X_test)

        return self.results(predictions, 'XGBoost')


    ##################################  MLP  ##################################
    def MLP(self):
        classifier = MLPClassifier()

        activation=['relu', 'tanh', 'identity', 'logistic']
        solver= ['adam', 'ibfgs', 'sgd']
        learning_rate=['constant', 'invscaling', 'adaptive']
        alpha=[0,.0001, 0.001, 0.01, 0.1, 1]
        max_iter = [500, 1000, 1500]

      # Create the param grid
        param_grid = {
                        'activation':activation,
                        'solver': solver,
                        'learning_rate': learning_rate,
                        'alpha': alpha,
                        'max_iter': max_iter
                     }


        clf = RandomizedSearchCV(classifier, param_grid, n_iter=50, cv=10)
        model = clf.fit(self.X_train, self.y_train)
  

        # make predictions
        optimized_model = model.best_estimator_
        pprint(optimized_model.get_params())
        predictions = optimized_model.predict(self.X_test)


        return self.results(predictions,  'MLP')

  