import pandas as pd

import pandas
from nltk.tag import StanfordPOSTagger
import nltk
import numpy as np

from sklearn import preprocessing

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
from sklearn.feature_extraction import DictVectorizer


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

def countVowels(string):
    vowel=("aıioöuüAEIİOÖUÜ")
    count=0
    for i in string:
        if i in vowel:
            count +=1
    return count

def countCons(string):
    cons=("bcçdfgğhjklmnprsştyzBCÇDFGĞHJKLMNPRSŞTVYZ")
    count=0
    for i in string:
        if i in cons:
            count +=1
    return count

    """
    For Stanford POS tagger
    String to Numeric value
    """

def score_to_numeric(x):
    if x=='CC':
        return 1
    elif x=='CD':
        return 2
    elif x=='DT':
        return 3
    elif x=='EX':
        return 4
    elif x=='FW':
        return 5
    elif x=='IN':
        return 6
    elif x=='JJ':
        return 7
    elif x=='JJR':
        return 8
    elif x=='JJS':
        return 9
    elif x=='LS':
        return 10
    elif x=='MD':
        return 11
    elif x=='NN':
        return 12
    elif x=='NNS':
        return 13
    elif x=='NNP':
        return 14
    elif x=='NNPS':
        return 15
    elif x=='PDT':
        return 16
    elif x=='POS':
        return 17
    elif x=='PRP':
        return 18
    elif x=='PRP$':
        return 19
    elif x=='RB':
        return 20
    elif x=='RBR':
        return 21
    elif x=='RBS':
        return 22
    elif x=='RP':
        return 23
    elif x=='SYM':
        return 24
    elif x=='TO':
        return 25
    elif x=='UH':
        return 26
    elif x=='VB':
        return 27
    elif x=='VBD':
        return 28
    elif x=='VBG':
        return 29
    elif x=='VBN':
        return 30
    elif x=='VBP':
        return 31
    elif x=='VBZ':
        return 32
    elif x=='WDT':
        return 33
    elif x=='WP':
        return 34
    elif x=='WP$':
        return 35
    elif x=='WRB':
        return 36
    elif x=='.':
        return 37
    elif x==':':
        return 38
    elif x==',':
        return 39
    elif x=='``':
        return 40
    elif x=='#':
        return 41
    elif x=='$':
        return 42
    elif x=='"':
        return 43
    elif x=='(':
        return 44
    elif x==')':
        return 45
    else:
        return 46


    """
    For NER Types
    String to Numeric value
    """

def NER_to_numeric(x):
    if (x == 'Person'):
        x = 1
    elif (x == 'Thing'):
        x = 2
    elif (x == 'Organization'):
        x = 3
    elif (x == 'Location'):
        x = 4
    elif (x == 'Product'):
        x = 5
    elif (x == 'Event'):
        x = 6
    elif (x == 'Character'):
        x = 7
    else:
        x = 0
    return x

def expFile(filename, POS_File):
    with open(filename) as file:
        lines = []
        index = 0
        set_tweets = set()
        for line in file:
            # The rstrip method gets rid of the "\n" at the end of each line
            lines.append(line.rstrip().split('\t'))
            index = index + 1
        NER_Value = []
        train_labels_dict = {}
        for line in lines:
            print(line)
            if (len(line) != 8 ):
                continue
            NER_Value = line[7].split(' ')
            train_labels_dict[line[0] + '_-_' + line[7]] = line[4]
            set_tweets.add(line[0] + '_-_' + line[1])
        print(train_labels_dict)
        print (set_tweets)

        words = []
        words.append('& Start Doc &')
        for letter in set_tweets:
            words.append('& Start Tweet &')
            index_set = 0
            tweet_id = ''

            #for word in letter.split():
            for word in nltk.word_tokenize(letter):
                if(index_set == 0):
                    tweet_word = word.split('_-_')
                    tweet_id = tweet_word[0]
                    words.append(tweet_id + '_-_' + tweet_word[1])
                else:
                    words.append(tweet_id + '_-_' + word)
                index_set = index_set + 1
            words.append('& End Tweet &')
        words.append('& End Doc &')
        print(words)

        POS_Taggers_dict = {}
        POS_lines = []
        with open(POS_File) as pos_file:
            for line in pos_file:
                POS_lines.append(line.rstrip().split('\t'))
            for line in POS_lines:
                POS_Taggers_dict[line[0]] = line[1]

        #Feature Set
        data = {
            #'1 WB Letter': [''],
            #'1 WB Is Capital': [''],
            #'1 WB Is All Capital': [''],
            #'1 WB Cons Vow Ratio': [''],
            '1 WB #': [''],
            '1 WB @': [''],

            #'2 W Letter': [''],
            '2 W Is Capital': [''],
            '2 W Is All Capital': [''],
            #'2 W Cons Vow Ratio': [''],
            '2 W POS Tag': ['']

            #'3 WA Letter': [''],
            #'3 WA Is Capital': [''],
            #'3 WA Is All Capital': [''],
            #'3 WA Cons Vow Ratio': ['']
        }

        df = pd.DataFrame(data, index=words)

        for j in range(df.index.size):

            """ Word Before """
            wordbefore=str(df.index[j - 1])
            if  (df.index[j - 1] == '& End Doc &' or df.index[j - 1] == '& Start Doc &' or df.index[j - 1] == '& Start Tweet &' or df.index[j - 1] == '& End Tweet &' or df.index[j - 1] == '& End Doc &' or df.index[j - 1] is None):
                #df.iloc[j]['1 WB Letter'] = 0
                #df.iloc[j]['1 WB Is Capital'] = 0
                #df.iloc[j]['1 WB Is All Capital'] = 0
                #df.iloc[j]['1 WB Cons Vow Ratio'] = 0
                df.iloc[j]['1 WB #'] = 0
                df.iloc[j]['1 WB @'] = 0
            else:
                #print('Word before: ')
                #print(df.index[j - 1])
                temp_index = df.index[j - 1]
                if "_-_"  in wordbefore:
                    temp_index = wordbefore.split('_-_')[1]
                #df.iloc[j]['1 WB Letter'] = len(temp_index)
                #df.iloc[j]['1 WB Is Capital'] = temp_index.istitle()
                #df.iloc[j]['1 WB Is All Capital'] = temp_index.isupper()
                #if (countVowels(temp_index) > 0):
                #    df.iloc[j]['1 WB Cons Vow Ratio'] = countCons(temp_index) / countVowels(temp_index)
                #else:
                #    df.iloc[j]['1 WB Cons Vow Ratio'] = 0
                if (temp_index == '#'):
                    df.iloc[j]['1 WB #'] = 1
                else:
                    df.iloc[j]['1 WB #'] = 0
                if (temp_index == '@'):
                    df.iloc[j]['1 WB @'] = 1
                else:
                    df.iloc[j]['1 WB @'] = 0

            """ Word """
            word_ = df.index[j]
            if (df.index[j] == '& Start Doc &' or df.index[j] == '& Start Tweet &' or df.index[j] == '& End Tweet &' or df.index[j] == '& End Doc &' or df.index[j] is None):
                #df.iloc[j]['2 W Letter'] = 0
                df.iloc[j]['2W Is Capital'] = 0
                df.iloc[j]['2 W Is All Capital'] = 0
                #df.iloc[j]['2 W Cons Vow Ratio'] = 0
                df.iloc[j]['2 W POS Tag'] = 0
            else:
                #print('Word: ')
                #print(df.index[j])
                temp_index = df.index[j]
                if "_-_" in word_:
                    temp_index = word_.split('_-_')[1]
                #df.iloc[j]['2 W Letter'] = len(temp_index)
                df.iloc[j]['2 W Is Capital'] = temp_index.istitle()
                df.iloc[j]['2 W Is All Capital'] = temp_index.isupper()
                #if (countVowels(df.index[j]) > 0):
                #    df.iloc[j]['2 W Cons Vow Ratio'] = countCons(temp_index) / countVowels(temp_index)
                #else:
                #    df.iloc[j]['2 W Cons Vow Ratio'] = 0
                df.iloc[j]['2 W POS Tag'] = score_to_numeric(POS_Taggers_dict.get(word_, ""))
            """ Word After """
            if (j+2 > df.index.size):
                break
            '''wordafter = df.index[j + 1]
            if (df.index[j+1] == '& Start Tweet &' or df.index[j+1] == '& End Tweet &' or df.index[j+1] == '& End Doc &' or df.index[j+1] is None or df.index[j] == '& Start Tweet &'):
                #df.iloc[j]['3 WA Letter'] = 0
                df.iloc[j]['3 WA Is Capital'] = 0
                df.iloc[j]['3 WA Is All Capital'] = 0
                #df.iloc[j]['3 WA Cons Vow Ratio'] = 0
            else:
                #print('Word after: ')
                #print(df.index[j + 1])
                temp_index = df.index[j+1]
                if "_-_" in wordafter:
                    temp_index = wordafter.split('_-_')[1]
                #df.iloc[j]['3 WA Letter'] = len(temp_index)
                df.iloc[j]['3 WA Is Capital'] = temp_index.istitle()
                df.iloc[j]['3 WA Is All Capital'] = temp_index.isupper()
                #if (countVowels(df.index[j + 1]) > 0):
                #    df.iloc[j]['3 WA Cons Vow Ratio'] = countCons(temp_index) / countVowels(temp_index)
                #else:
                #    df.iloc[j]['3 WA Cons Vow Ratio'] = 0
                '''
        df = df.drop(['& Start Doc &','& Start Tweet &','& End Tweet &','& End Doc &'])

        print('df - remove oncesi')
        print(df)
        df_temp = []
        for m in range(df.index.size):
            # Remove Type 0 - BEG
            word_remove = df.index[m]
            value_remove = train_labels_dict.get(word_remove, "")
            if (value_remove == ''):
                df_temp.append(df.index[m])
            # Remove Type 0 - END
        df = df.drop(df_temp)
        print('df - remove sonrasi')
        print(df)

        df_train_test_arrays = df.astype(int)
        print(df_train_test_arrays)
        train_test_labels_array = []
        words_temp = []

        thefile = open('NERTypes.txt', 'w')
        for index in df.iterrows():
            temp_word = str(index[0])
            if('_-_@' in temp_word ):
                temp_word = temp_word.replace('_-_@','_-_')
            elif('_-_#' in temp_word ):
                temp_word = temp_word.replace('_-_#','_-_')
            value_d = train_labels_dict.get(temp_word, "")
            if (value_d != ''):
                thefile.write("%s\n" % value_d)
            else:
                continue

            value_d = NER_to_numeric(value_d)

            train_test_labels_array.append(value_d)
            words_temp.append(temp_word)
        return_df_array = [df_train_test_arrays,train_test_labels_array]
        return return_df_array

import os
"""import Vectorizer"""
import string
from sklearn import svm
"""from Estimation import Estimation"""
import time
import configparser
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
def main():

    # Train File
    print('Train File Starting')
    df_train = expFile('microposts2016-neel-training_neel7.txt', 'POS_Train.txt')
    print('Train File Ended')
    df_Xtrain = df_train[0]
    df_Xtrain_array = df_Xtrain.values
    X = df_Xtrain_array[:, 0:5]
    Y = df_train[1]

    clf_lr = LogisticRegression()
    clf_lr.fit(X, Y)
    # /Train File

    # Test File
    print('Test File Starting')
    df_test = expFile('microposts2016-neel-test_neel7.txt','POS_Test.txt')
    print('Test File Ended')
    df_Xtest = df_test[0]
    df_Xtest_array = df_Xtest.values
    X = df_Xtest_array[:, 0:5]
    Y = df_test[1]
    # /Test File

    predictions = clf_lr.predict(X)
    print('Accuracy Score')
    print(accuracy_score(Y, predictions))
    print('Confusion Matrix')
    print(confusion_matrix(Y, predictions))
    print('Classification Report')
    print(classification_report(Y, predictions))
    print('F1 micro average')
    print(f1_score(Y, predictions, average='micro'))

if __name__ == "__main__":
    main()