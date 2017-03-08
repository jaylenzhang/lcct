#!usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import os
import time
reload(sys)
sys.setdefaultencoding('utf8')
import ConfigParser
import logging
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
import numpy as np
import cPickle as pickle

HAS_XGB = True
try:
    import xgboost as xgb
except:
    HAS_XGB=False

HAS_LGBM = True
try:
    import lightgbm as lgbm
except:
    HAS_LGBM = False

def classify(model, test_x, thresh=0.5):
    proba = model.predict_proba(test_x)
    predict = [1 if pro[1] >= thresh else -1 for pro in proba]
    return predict


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2',n_jobs=-1)
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50,n_jobs=-1)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    svr = OneVsRestClassifier(SVC(kernel='rbf', probability=True),n_jobs=-1)
    #model = SVC(kernel='rbf', probability=True)
    model = svr
    model.fit(train_x, train_y)
    return model


def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print para, val
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model
def xgboost_classifier(train_x,train_y):
    clf = xgb.XGBClassifier(n_estimators = 100,nthread=10,max_depth=5,objective= 'multi:softprob')
    #print clf.attributes()
    clf.fit(train_x, train_y)
    return clf
def lightgbm_classifer(train_x,train_y):
    clf = lgbm.LGBMRegressor(objective='binary',num_leaves=31,learning_rate=0.05,n_estimators=100,nthread=10)
    clf.fit(train_x, train_y)
    return clf
def get_leaf_feature(leaf,n_estimators,max_depth):
    feature_len =np.power(2, max_depth+1)
    train_x = []
    for sample in leaf:
        sample_x = [0 for i in range(feature_len*n_estimators)]
        for k in range(n_estimators):
            leaf_index = sample[k]
            index = k*feature_len + leaf_index
            sample_x[index] = 1
        train_x.append(sample_x)
    return np.array(train_x)

class  Classifer():
    def __init__(self,conf_in):
        self.cf = ConfigParser.ConfigParser()
        self.cf.read(conf_in)
        #basic_conf
        self.work_dir = self.cf.get('basic_conf','work_dir')
        self.data_dir = self.work_dir + './data/'
        self.conf_dir = self.work_dir + './conf/'
        self.stopword_data = self.conf_dir + 'stopwords.txt'
        
        self.word_min_count =  self.cf.getint('basic_conf','word_min_count')
        self.label2clsName_data = self.data_dir + self.cf.get('basic_conf','label2clsName')
        self.id2docName_data = self.data_dir + self.cf.get('basic_conf','id2docName')
        self.word2id_data = self.data_dir + self.cf.get('basic_conf','word2id')
        self.tfidf_svm_data = self.data_dir + self.cf.get('basic_conf','tfidf_svm')
        self.word2idf_data = self.data_dir + self.cf.get('basic_conf','word2idf')
        self.train_test_dir = self.data_dir + self.cf.get('basic_conf','train_test_dir')
        if not os.path.exists(self.train_test_dir) :
            os.mkdir(self.train_test_dir)
        self.cross_validation_num = self.cf.getint('pre_process','cross_validation_num')
        # set loging 
        ISOTIMEFORMAT='%Y%m%d-%H%M%S'
        time_str = 'pre_process-'+ time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=self.work_dir + '/log/log-' + time_str + '.txt',filemode='w')
        self.logging = logging
    def classifier(self):
        model_save = {}
        target_names = []
        with open(self.label2clsName_data,'r') as fin:
            target_names_tmp = []
            for line in fin:
                line = line.rstrip('\n')
                line_sp = line.split('\t')
                target_names_tmp.append((int(line_sp[0]),line_sp[1]))
            target_names_tmp = sorted(target_names_tmp,key=lambda x:x[0],reverse=False)
            for id, label in target_names_tmp:
                target_names.append(label)
        
        test_classifiers     = ['NB', 'KNN', 'LR', 'RF', 'DT'] #,'GBDT'] #,'SVM']
        ensemble_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT'] #,'GBDT'] #,'SVM']
        if HAS_LGBM :
            pass
            #test_classifiers     = ['LIGHTGBM'] + test_classifiers
            #ensemble_classifiers     = ['LIGHTGBM'] + ensemble_classifiers
        if HAS_XGB :
            pass
            #test_classifiers     = ['XGBOOST'] + test_classifiers
            #ensemble_classifiers     = ['XGBOOST'] + ensemble_classifiers
            
        #ensemble_classifiers = ['KNN', 'LR', 'RF', 'SVM', 'GBDT']
        
        classifiers = {'NB':naive_bayes_classifier, 
                      'KNN':knn_classifier,
                       'LR':logistic_regression_classifier,
                       'RF':random_forest_classifier,
                       'DT':decision_tree_classifier,
                      'SVM':svm_classifier,
                    'SVMCV':svm_cross_validation,
                     'GBDT':gradient_boosting_classifier,
                     'XGBOOST':xgboost_classifier,
                     'LIGHTGBM':lightgbm_classifer
        }
        
        evalue_dict = dict()
        for i in range(self.cross_validation_num):
            print 'reading part [%d] training and testing data...'%(i)
            train_file = self.train_test_dir + '/train_resample_' + str(i) + '.svm'
            test_file = self.train_test_dir + '/test_resample_' + str(i) + '.svm'
            train_x, train_y = load_svmlight_file(train_file,dtype=np.float64)
            test_x, test_y   = load_svmlight_file(test_file,dtype=np.float64)
            num_train, num_feat = train_x.shape
            num_test, num_feat = test_x.shape
            print '******************** Data Info *********************'
            print '#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)
            if i not in model_save:
                model_save[i] = dict()
            for classifier in test_classifiers:
                print '******************* %s ********************' % classifier
                sys.stdout.flush()
                if classifier not in evalue_dict:
                    evalue_dict[classifier] = dict()
                    evalue_dict[classifier]['acc'] = []
                    evalue_dict[classifier]['precision'] = []
                    evalue_dict[classifier]['recall'] = []
                    evalue_dict[classifier]['fscore'] = []
                    evalue_dict[classifier]['time'] = []
                t0 = time.time()
                model = classifiers[classifier](train_x, train_y)
                train_time = time.time() - t0
                predict = model.predict(test_x)
                model_save[i][classifier] = model

                acc = metrics.accuracy_score(test_y, predict) 
                precision = metrics.precision_score(test_y, predict,average='micro')
                recall = metrics.recall_score(test_y, predict,average='micro')
                fscore = metrics.f1_score(test_y, predict,average='micro')
                evalue_dict[classifier]['acc'].append(acc)
                evalue_dict[classifier]['precision'].append(precision)
                evalue_dict[classifier]['recall'].append(recall)
                evalue_dict[classifier]['fscore'].append(fscore)
                evalue_dict[classifier]['time'].append(train_time)

                print 'acc:%.2f%%' % (100*acc) 
                print metrics.classification_report(test_y, predict,target_names=target_names)
        fmt='%12s%12s%12s%12s%12s%12s' 
        print fmt%('classifier','acc','precision','recall','fscore','train_time')
        for classifier in test_classifiers:
            acc = np.mean(evalue_dict[classifier]['acc'])
            precision = np.mean(evalue_dict[classifier]['precision'])
            recall = np.mean(evalue_dict[classifier]['recall'])
            fscore = np.mean(evalue_dict[classifier]['fscore'])
            train_time = np.mean(evalue_dict[classifier]['time'])

            print fmt% tuple( [classifier] + [str('%.3f'%x) for x in [acc,precision,recall,fscore,train_time] ])
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python ' + sys.argv[0] + 'conf_in'
        sys.exit(1)
    cls = Classifer(sys.argv[1])
    cls.classifier()
