import pandas

class MyEvaluation(object):
    def __init__(self):
        self._accuracy = 0.0
        self._AUC = 0.0
        self._macroF1 = 0.0
        self._microF1 = 0.0
        self._F1 = []
        self._Precision = []
        self._Recall = []
        self._Conf_Matrix= pandas.DataFrame

    def _getAccuracy(self):
        return self._accuracy

    def _setAccuracy(self,value):
        self._accuracy=value

    def _getAUC(self):
        return self._AUC

    def _setAUC(self, value):
        self._AUC = value

        return self._Precision

    def _getPrecision(self):
        return self._Precision

    def _setPrecision(self, value):
        self._Precision = value

    def _getRecall(self):
        return self._Recall

    def _setRecall(self, value):
        self._Recall = value

    def _getF1(self):
        return self._F1

    def _setF1(self, value):
        self._F1 = value

    def _getF1_macro(self):
        return self._F1_macro

    def _setF1_macro(self, value):
        self._F1_macro = value

    def _getF1_micro(self):
        return self._F1_micro

    def _setF1_micro(self, value):
        self._F1_micro = value

    def _getConfusionMatrix(self):
        return self._Conf_Matrix

    def _setConfusionMatrix(self, value):
        self._Conf_Matrix = value


from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
"""from Evaluation import MyEvaluation
"""

class Estimation(object):
    def __init__(self):
        return

    def evaluate_testOnTrainingSet(self,dataset,classlabels,classifier):
        predictions = classifier.predict(dataset)
        accuracy = metrics.accuracy_score(classlabels, predictions)
        precision = precision_score(classlabels, predictions, average=None)
        recall = recall_score(classlabels, predictions, average=None)

        F1 = metrics.f1_score(classlabels, predictions, average=None)
        F1_macro = metrics.f1_score(classlabels, predictions, average='macro')
        F1_micro = metrics.f1_score(classlabels, predictions, average='micro')

        cm = confusion_matrix(classlabels, predictions)

        myEval = MyEvaluation()
        myEval._setAccuracy(accuracy.mean())
        myEval._setPrecision(precision)
        myEval._setRecall(recall)
        myEval._setF1(F1)
        myEval._setF1_macro(F1_macro)
        myEval._setF1_micro(F1_micro)
        myEval._setConfusionMatrix(cm)

        return myEval



    def evaluate_nFoldCV(self,dataset,classlabels,classifier,n):
        # k_fold = KFold(len(y_test), n_folds=n, shuffle=True, random_state=0)
        predictions=cross_validation.cross_val_predict(classifier,dataset,classlabels,cv=n)
        # score=cross_val_score(classifier, dataset, classlabels,cv=n,scoring='accuracy')  avg is same as metrics.accuracy_score

        accuracy = metrics.accuracy_score(classlabels, predictions)
        precision = precision_score(classlabels, predictions, average=None)
        recall = recall_score(classlabels, predictions, average=None)

        F1=metrics.f1_score(classlabels, predictions, average=None)
        F1_macro=metrics.f1_score(classlabels, predictions, average='macro')
        F1_micro=metrics.f1_score(classlabels, predictions, average='micro')

        cm = confusion_matrix(classlabels, predictions)

        myEval = MyEvaluation()
        myEval._setAccuracy(accuracy.mean())
        myEval._setPrecision(precision)
        myEval._setRecall(recall)
        myEval._setF1(F1)
        myEval._setF1_macro(F1_macro)
        myEval._setF1_micro(F1_micro)
        myEval._setConfusionMatrix(cm)
        return myEval



    def evaluate_trainTestSplit(self,dataset,classLabels,classifier,testPercantage):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataset, classLabels, test_size=testPercantage,random_state=0)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = classifier.score(X_test, y_test)
        precision = precision_score(y_test, predictions, average=None)
        recall = recall_score(y_test, predictions, average=None)

        F1=metrics.f1_score(y_test, predictions, average=None)
        F1_macro=metrics.f1_score(y_test, predictions, average='macro')
        F1_micro=metrics.f1_score(y_test, predictions, average='micro')
        # auc = roc_auc_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        myEval = MyEvaluation()
        myEval._setAccuracy(accuracy)
        myEval._setPrecision(precision)
        myEval._setRecall(recall)
        myEval._setF1(F1)
        myEval._setF1_macro(F1_macro)
        myEval._setF1_micro(F1_micro)
        myEval._setConfusionMatrix(cm)
        # myEval._setAUC(auc)

        return myEval

import os
"""import Vectorizer"""
import string
from sklearn import svm
"""from Estimation import Estimation"""
import time
import configparser
from sklearn.linear_model import LogisticRegression

def main():
    print('Hello')
    estimation=Estimation()
    start = time.time()
    classifier = svm.SVC(kernel='linear',C=1.0,decision_function_shape=None)
    start = time.time()
    classifier_model = classifier.fit(train_arrays,train_labels)
    myEval_nFoldCV_SVM=estimation.evaluate_nFoldCV(test_array,test_labels,classifier_model,n)
    end=time.time()

    print("\n -----------------" , n ,"fold CV with SVM / DOC2VEC------------------------ \n")
    print("Accuracy :", myEval_nFoldCV_SVM._getAccuracy())
    print("Precision:", myEval_nFoldCV_SVM._getPrecision())
    print("Recall   :", myEval_nFoldCV_SVM._getRecall())
    print("F1       :" , myEval_nFoldCV_SVM._getF1())
    print("F1 macro :" , myEval_nFoldCV_SVM._getF1_macro())
    print("F1 micro :" , myEval_nFoldCV_SVM._getF1_micro())
    print("Confusion Matrix :\n " , myEval_nFoldCV_SVM._getConfusionMatrix())
    print("Time : ", end - start)

if __name__ == "__main__":
    main()