#!/usr/bin/env python
# -*- coding: utf-8 -*-  
# @author Zhang zhiming (zhangzhiming@)
# date

import re
import json
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
import ConfigParser
import logging
import time
import math
import random
import os
import numpy as np 
class TF_v1():
    def __init__(self):
        pass
    def tf(self,word_dict):
        return word_dict
class TF_v2():
    def __init__(self):
        pass
    def tf(self,word_dict):
        tf_ = dict()
        for w in word_dict:
            tf_[w] = 1 + math.log(word_dict[w])
        return tf_
class TF_v3():
    def __init__(self):
        pass
    def tf(self,word_dict):
        tf_ = dict()
        max_count = float(max(word_dict.values()))
        alph = 0.4
        for w in word_dict:
            tf_[w] = alph + (1-alph) * (word_dict[w]/max_count)
        return tf_
class TFFunc():
    def __init__(self,func):
        self.func = func
        pass
    def tf(self,word_dict):
        return self.func.tf(word_dict)

class IDF_v1():
    def __init__(self):
        pass
    def idf(self,idf_dict,doc_num):
        idf_ = dict()
        for w in idf_dict:
            idf_[w] = math.log(doc_num*1.0/idf_dict[w])
        return idf_
class IDF_v2():
    def __init__(self):
        pass
    def idf(self,idf_dict,doc_num):
        idf_ = dict()
        for w in idf_dict:
            idf_[w] = math.log(doc_num/(1.0+idf_dict[w]))
        return idf_
class IDFFunc():
    def  __init__(self,func):
        self.func = func
    def idf(self,idf_dict,doc_num):
        return self.func.idf(idf_dict,doc_num)
##########################################
#
#process begin 
##########################################
class PreProcess():
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
        self.file_tag = self.cf.get('pre_process','file_tag')
        self.json_data = self.data_dir + self.cf.get('pre_process','json_data')
        self.wordseg_data = self.data_dir + self.cf.get('pre_process','wordseg_data')
        self.vocab_data = self.data_dir + self.cf.get('pre_process','vocab_data')
        self.tfidf_data = self.data_dir + self.cf.get('pre_process','tfidf_data')
        self.train_test_rate = self.cf.getfloat('pre_process','train_test_rate')
        self.cross_validation_num = self.cf.getint('pre_process','cross_validation_num')
        self.ignore_df_rate = self.cf.getfloat('pre_process','ignore_df_rate')
        # set loging 
        ISOTIMEFORMAT='%Y%m%d-%H%M%S'
        time_str = 'pre_process-'+ time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=self.work_dir + '/log/log-' + time_str + '.txt',filemode='w')
        self.logging = logging
        #self.Log(' '.join(['[init]','done']))
    def get_stopword(self):
        self.stopword_set = set()
        with open(self.stopword_data,'r') as fin:
            for line in fin:
                line = u''+line.strip()
                self.stopword_set.add(line)


    def generate_voc(self):
        idf_count = dict()
        doc_num = 0
        with open(self.wordseg_data,'r') as fin:
            for line in fin:
                line = line.rstrip('\n')
                line_sp = line.split('\t')
                doc_json = json.loads(line_sp[1])
                content_seg_list = doc_json['content_seg_list']
                word_set = set()
                for sent in content_seg_list:
                    for w in sent:
                        if len(w.strip()) == 0:
                            continue
                        word_set.add(w)
                for w in word_set:
                    idf_count[w] = idf_count.get(w,0) + 1
                doc_num += 1
        ignore_word_set = set()
        for w in idf_count:
            df = idf_count[w]*1.0 / doc_num
            if df > self.ignore_df_rate:
                ignore_word_set.add(w)
        print >> sys.stderr, '[ignore_word_set]',len(ignore_word_set),' '.join(ignore_word_set)
        word_dict = dict()
        self.get_stopword()
        with open(self.wordseg_data,'r') as fin:
            for line in fin:
                line = u''+line.rstrip('\n')
                line_sp = line.split('\t')
                doc_json = json.loads(line_sp[1])
                content_seg_list = doc_json['content_seg_list']
                for sent in content_seg_list:
                    for w in sent:
                        if len(w.strip()) == 0:
                            continue
                        if w in self.stopword_set:
                            continue
                        if w in ignore_word_set:
                            continue
                        word_dict[w] = word_dict.get(w,0) + 1
            word_list = sorted(word_dict.items(),key=lambda x:x[1],reverse=True)
        self.vocab2freq = dict()
        with open(self.vocab_data,'w') as fout:
            for w ,c in word_list:
                if c <= self.word_min_count:
                    continue
                fout.write( '\t'.join([w,str(c)]) + '\n')
                self.vocab2freq[w] = c

        self.Log(' '.join(['[generate_voc]','done']))
    def get_voc(self):
        self.vocab2freq = dict()
        self.word2id = dict()
        self.id2word = dict()
        if not os.path.exists(self.vocab_data) :
            self.Log(' '.join(['[get_voc]','vocab_dat not exists ']) )
            sys.exit(1)
        index = 1
        with open(self.vocab_data,'r') as fin:
            for line in fin:
                line = u''+line.rstrip('\n')
                line_sp = line.split('\t')
                w,c  = line_sp
                self.vocab2freq[w] = c
                self.word2id[w] = index
                self.id2word[index] = w
                index += 1
        with open(self.word2id_data,'w') as fout:
            for i in range(len(self.id2word)):
                i = i+1
                fout.write('\t'.join([self.id2word[i],str(i)]) + '\n')

    def get_tfidf(self,tf_func,idf_func):
        self.get_voc()
        self.get_stopword()

        self.idf = dict()
        doc_list = []
        # 1. idf
        idf_count = dict()
        doc_num = 0
        class_type_dict = dict()
        with open(self.wordseg_data,'r') as fin:
            for line in fin:
                line = line.rstrip('\n')
                line_sp = line.split('\t')
                doc_json = json.loads(line_sp[1])
                content_seg_list = doc_json['content_seg_list']
                word_set = set()
                for sent in content_seg_list:
                    for w in sent:
                        if len(w.strip()) == 0:
                            continue
                        if w in self.stopword_set:
                            continue
                        if w not in self.word2id:
                            continue
                        word_set.add(w)
                for w in word_set:
                    idf_count[w] = idf_count.get(w,0) + 1
                doc_num += 1
                cls_name = doc_json['cls_name']
                class_type_dict[cls_name] = class_type_dict.get(cls_name,0) + 1
        idf = idf_func.idf(idf_count,doc_num)
        cnt = -0
        for w in idf_count:
            print '\t'.join([w,str(idf_count[w]),str(idf[w])])
            if cnt == 10:
                break
            cnt += 1
        # 2. asign class_type id 
        cls_type_list = sorted(class_type_dict.items(),key=lambda x:x[1],reverse=True)
        #print '\n'.join([k[0]+"\t"+str(k[1]) for k in cls_type_list])
        cls_id = 0
        
        for cls,cnt in cls_type_list:
            class_type_dict[cls] = str(cls_id)
            cls_id += 1
        # 3. compute & write tfidf ,write svm
        doc_id = 0
        doc_id_list = []
        vocab_len = len(self.word2id)
        with open(self.wordseg_data,'r') as fin,open(self.tfidf_svm_data,'w') as fout,open(self.tfidf_data,'w') as fout_tfidf:
            for line in fin:
                line = line.rstrip('\n')
                line_sp = line.split('\t')
                doc_json = json.loads(line_sp[1])
                content_seg_list = doc_json['content_seg_list']
                word_dict = dict()
                url = doc_json['url']
                cls_name = doc_json['cls_name']
                for sent in content_seg_list:
                    for w in sent:
                        if len(w.strip()) == 0:
                            continue
                        if w in self.stopword_set:
                            continue
                        if w not in self.vocab2freq:
                            continue
                        word_dict[w] = word_dict.get(w,0) + 1
                #tfidf
                tf = tf_func.tf(word_dict)
                tfidf = dict()
                word_tfidf = dict()
                for w in tf:
                    w_id = self.word2id[w]
                    if idf[w] < math.log(2):
                        continue
                        pass
                        #print '\t'.join(['tfid=0',w,str(w_id),str(tf[w]),str(idf[w])])
                        #continue
                    tfidf[w_id]  = tf[w] * idf[w]
                    word_tfidf[w] = tfidf[w_id]
                # svm file 
                vec = []
                vec.append(class_type_dict[cls_name])
                tfidf_list = sorted(tfidf.items(),key=lambda x:x[0],reverse=False)
                
                for i,tfidf_ in tfidf_list:
                    vec.append(str(i) +':'+str('%.6f'%tfidf_))
                if len(tfidf) != 0 and tfidf_list[-1][0] != vocab_len+1:
                    i = vocab_len+1
                    vec.append(str(i) +':'+str('%.6f'%0))
                    
                fout.write(' '.join(vec) + '\n')
                doc_id_list.append([url,cls_name,vec])
                doc_id += 1
                
                #tfidf file 
                doc_json.pop('content_seg_list')
                word_tfidf_list = sorted(word_tfidf.items(),key=lambda x:x[1] ,reverse=True)
                doc_json['tfidf'] = word_tfidf_list
                fout_tfidf.write('\t'.join([line_sp[0],json.dumps(doc_json,ensure_ascii=False) ]) + '\n')
        # 4. label2clsName_data
        with open(self.label2clsName_data,'w') as fout:
            class_type_list = sorted(class_type_dict.items(),key=lambda x:x[1],reverse=True)
            for cls,cnt in class_type_list:
                cls_id = class_type_dict[cls] 
                fout.write('\t'.join([str(cls_id),cls]) + '\n')
        # 5. id2docName_data
        doc_id = 0
        with open(self.id2docName_data,'w') as fout:
            for i in range(len(doc_id_list)):
                fout.write('\t'.join([str(i)] + doc_id_list[i][:2]) + '\n')
        # 6. word2idf
        with open(self.word2idf_data,'w') as fout:
            idf_list = sorted(idf.items(),key = lambda x:x[1],reverse=True)
            for w,v in idf_list:
                fout.write('\t'.join([w,str(v)]) + '\n')
        
        # 7. split train,test
        
        cls_name_count_dict = dict()
        cls_name_data_dict = dict()
        for url,cls_name,vec in doc_id_list:
            cls_name_count_dict[cls_name] = cls_name_count_dict.get(cls_name,0) + 1
            if cls_name not  in cls_name_data_dict:
                cls_name_data_dict[cls_name] = []
            cls_name_data_dict[cls_name].append((url,cls_name,vec))
        # max_cls_data_size upbound of resample size per cls_name 
        max_cls_data_size = max(cls_name_count_dict.values())
        train_samples = int( max_cls_data_size * self.train_test_rate)
        for k in range(self.cross_validation_num):
            train_result = []
            test_result = []
            # resample train data ,ensure every cls_name has the same train_size 
            for cls_name in cls_name_data_dict:
                cls_name_len = len(cls_name_data_dict[cls_name])
                cls_name_list = range(cls_name_len)
                random.shuffle(cls_name_list)

                cls_name_train_samples = int( cls_name_len * self.train_test_rate)
                cls_name_test_samples = cls_name_len - cls_name_train_samples
                cls_name_train_result = []
                for i in range(train_samples):
                    idx = random.choice(cls_name_list[:cls_name_train_samples])
                    train_result.append(cls_name_data_dict[cls_name][idx])
                for idx in cls_name_list[cls_name_train_samples:]:
                    test_result.append( cls_name_data_dict[cls_name][idx])
            
            train_name = self.train_test_dir+'/train_resample_' +str(k)
            ftrain = open(train_name + '.svm','w')
            ftrain_map = open(train_name +'_map.txt','w')
            for url,cls_name,vec in train_result:
                ftrain.write(' '.join(vec) + '\n')
                ftrain_map.write('\t'.join([url,cls_name]) + '\n')
            test_name = self.train_test_dir+'/test_resample_' +str(k)
            ftest = open(test_name + '.svm','w')
            ftest_map = open(test_name +'_map.txt','w')
            for url,cls_name,vec in test_result:
                ftest.write(' '.join(vec) + '\n')
                ftest_map.write('\t'.join([url,cls_name]) + '\n')
            ftrain.close()
            ftrain_map.close()
            ftest.close()
            ftest_map.close()




        # 8. split train,test
        train_samples = int(doc_num * self.train_test_rate)
        for k in range(self.cross_validation_num):
            doc_num_list = range(len(doc_id_list))
            random.shuffle(doc_num_list)
            train_name = self.train_test_dir+'/train_' +str(k)
            ftrain = open(train_name + '.svm','w')
            ftrain_map = open(train_name +'_map.txt','w')
            for i in doc_num_list[:train_samples]:
                url,cls_name,vec = doc_id_list[i]
                ftrain.write(' '.join(vec) + '\n')
                ftrain_map.write('\t'.join([str(i)] + doc_id_list[i][:2]) + '\n')
            test_name = self.train_test_dir+'/test_' +str(k)
            ftest = open(test_name + '.svm','w')
            ftest_map = open(test_name +'_map.txt','w')
            for i in doc_num_list[train_samples:]:
                url,cls_name,vec = doc_id_list[i]
                ftest.write(' '.join(vec) + '\n')
                ftest_map.write('\t'.join([str(i)] + doc_id_list[i][:2]) + '\n')

            ftrain.close()
            ftrain_map.close()
            ftest.close()
            ftest_map.close()

    def LogTemplate(self, s):
        time_stmap =  ''
        return  ' [' + time_stmap + ']:  ' + str(s)
    def Log(self, s):
        ss =  self.LogTemplate(s)
        self.logging.info(ss)
    def LogErr(self, s):
        ss =  self.LogTemplate(s)
        self.logging.error(ss)
def test(conf_in):
    pre = PreProcess(conf_in)
    pre.generate_voc()
    pre.get_voc()
    tf_v1 = TF_v1()
    idf_v1 = IDF_v1()
    tf = TFFunc(tf_v1)
    idf = IDFFunc(idf_v1)
    pre.get_tfidf(tf,idf)
if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: pre_proccess_tfidf.py \n')
    test(sys.argv[1])
