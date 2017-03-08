#!/bin/sh

CUR_DIR=`pwd`
cd ../src/

python classify.py ../conf/lcct.cfg > ../log/classify.log_chi2 2>&1 &

exit 0

