
"""
@author: yingying
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import uniform, truncnorm, randint
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
        logistic_regression = classifier.fit(self.X_train, self.y_train)
        # make predictions
        predictions = logistic_regression.predict(self.X_test)

        return self.results(predictions, 'Logistic Regression')

  ###########################  Naive Bayes Classifier ########################
    def NaiveBayes(self):
        classifier = GaussianNB()
        naive_bayes = classifier.fit(self.X_train, self.y_train)

        # make predictions
        predictions = naive_bayes.predict(self.X_test)

        return self.results(predictions, 'Naive Bayes')

    ##############################  Decision Tree #############################
    def DecisionTree(self):
        # classifier = DecisionTreeClassifier(random_state=0)
        classifier = DecisionTreeClassifier()
        decision_tree = classifier.fit(self.X_train, self.y_train)

        # make predictions
        predictions = decision_tree.predict(self.X_test)

        return self.results(predictions, 'Decision Tree')

    ##############################  Random Forest ##############################

    def RandomForest(self):
        classifier = RandomForestClassifier()
        random_forest = classifier.fit(self.X_train, self.y_train)

        # make predictions
        predictions = random_forest.predict(self.X_test)

        return self.results(predictions,  'Random Forest')

    ############################  KNearestNeighbours ############################
    def KNearestNeighbours(self):
        # classifier = KNeighborsClassifier(n_neighbors=30)
        classifier = KNeighborsClassifier()
        knn = classifier.fit(self.X_train, self.y_train)

        # make predictions
        predictions = knn.predict(self.X_test)

        return self.results(predictions,  'K Nearest Neighbours')

    ##################################  SVM ##################################
    def SVM(self):
        # classifier = SVC(kernel='poly')
        classifier = SVC()
        svc = classifier.fit(self.X_train, self.y_train)

        # make predictions
        predictions = svc.predict(self.X_test)

        return self.results(predictions,  'SVC')

    #################################  XGBoost ################################
    def XGBoost(self):
        classifier = xgboost.XGBClassifier()
        # xg_boost = classifier.fit(
        #     self.X_train, self.y_train, eval_metric='rmse')
        xg_boost = classifier.fit(
            self.X_train, self.y_train)

        # make predictions
        predictions = xg_boost.predict(self.X_test)

        return self.results(predictions,  'XGBoost')

    #################################  AdaBoost  ##############################
    def AdaBoost(self):
        classifier = AdaBoostClassifier()
        ada = classifier.fit(self.X_train, self.y_train)

        # make predictions
        predictions = ada.predict(self.X_test)

        return self.results(predictions,  'Ada Boost')

    ##################################  MLP  ##################################
    def MLP(self):
        # classifier = MLPClassifier(alpha=1, max_iter=1000)
        classifier = MLPClassifier()
        mlp = classifier.fit(self.X_train, self.y_train)

        # make predictions
        predictions = mlp.predict(self.X_test)

        return self.results(predictions,  'MLP')

  ###############################  HyperTuning ##############################
    # def RandomizedSearch(self):

    #     classifier = xgboost.XGBClassifier()

    #     n_estimators = [50, 100, 250, 500, 750, 1000, 1500]
    #     # max_depth = [2,3,5,10,15]
    #     booster = ['gbtree', 'gblinear']
    #     base_score = [0.25, 0.5, 0.75, 1]
    #     # min_child_weight =[1,2,3,4,5]
    #     learning_rate = [0.05, 0.1, 0.15, 0.20]

    #     hyperparameter_grid = {
    #         'n_estimators': n_estimators,
    #         # 'max_depth':max_depth,
    #         'booster': booster,
    #         # 'min_child_weight':min_child_weight,
    #         'learning_rate': learning_rate
    #     }

    #     result = RandomizedSearchCV(estimator=classifier, param_distributions=hyperparameter_grid, n_iter=50, cv=3,
    #                                 random_state=42, verbose=0)
    #     result.fit(self.X_train, self.y_train, eval_metric='rmse')

    #     xgrf2 = result.best_estimator_
    #     predictions = xgrf2.predict(self.X_test)

    #     return self.results(predictions, 'Randomized Search')
