#!/usr/bin/env python
# -*- coding: utf-8 -*-  
# @author Zhang zhiming (zhangzhiming@)
# date

import re
import json
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )


def select_type(data_in):
    cls_name_dict = dict()
    with open(data_in,'r') as fin:
        for line in fin:
            line = line.rstrip('\n')
            line_sp = line.split('\t')
            doc_json = json.loads(line_sp[1])
            cls_name = doc_json['cls_name']
            cls_name_dict[cls_name] = cls_name_dict.get(cls_name,0) + 1
    cls_name_list = sorted(cls_name_dict.items(),key=lambda x:x[1],reverse=True)
    #print '\n'.join(['\t'.join([str(x) for x in k]) for k in cls_name_list])
    with open(data_in,'r') as fin:
        for line in fin:
            line = line.rstrip('\n')
            line_sp = line.split('\t')
            doc_json = json.loads(line_sp[1])
            cls_name = doc_json['cls_name']
            if cls_name_dict[cls_name] > 400:
                print line

    #cls_name_list = sorted(cls_name_dict.items(),key=lambda x:x[1],reverse=True)
if __name__ == '__main__':
    if len(sys.argv) !=2:
        sys.stderr.write('Usage: select_type.py data_in \n')
        sys.exit(1)
    select_type(sys.argv[1])
