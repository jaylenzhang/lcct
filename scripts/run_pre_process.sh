#!/bin/sh

CUR_DIR=`pwd`
echo $CUR_DIR
cd ../src

# 1. merge train and test file 
cat ../data/fudan_train.json ../data/fudan_train.json  > ../data/classify.json

# 2. jieba word seg
cat ../data/classify.json | python jieba_for_classify.py > ../data/classify.seg

# 3. preprocess and generate libsvm file to train 
python pre_proccess_tfidf.py ../conf/lcct.cfg

cd $CUR_DIR

exit 0

