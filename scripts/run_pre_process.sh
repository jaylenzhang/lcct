#!/bin/sh

CUR_DIR=`pwd`
echo $CUR_DIR
cd ../src

cat ../data/fudan_train.json ../data/fudan_train.json  > ../data/classify.select
#python select_type.py  ../data/classify.select >  ../data/classify.json
cat  ../data/classify.select >  ../data/classify.json
cat ../data/classify.json | python jieba_for_classify.py > ../data/classify.seg
python pre_proccess_tfidf.py ../conf/lcct.cfg

cd $CUR_DIR

exit 0

