#!/usr/bin/env python
# -*- coding: utf-8 -*-  
# @author Zhang zhiming (zhangzhiming@)
# date

import re
import json
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
import jieba
jieba.enable_parallel(4)
jieba.set_dictionary('../conf/dict.txt.big')
jieba.initialize()
#abc = "公安部治安局局长刘绍武介绍，这次销毁的非法枪支来源于三个方面。刘绍武：打击破案包括涉黑、涉恶的团伙犯罪、毒品犯罪，还有从境外非法走私的枪支爆炸物。"
#print json.dumps(jieba.lcut(abc),ensure_ascii=False)
for line in sys.stdin:
    line = line.rstrip('\n')
    line_sp = line.split('\t')
    xml_json = json.loads(line_sp[1])
    content_seg_list = []
    if 'content_list' in xml_json:
        content_list = xml_json['content_list']
        for c in content_list:
            content_seg = jieba.lcut(c)
            content_seg_list.append(content_seg)
    xml_json['content_seg_list'] = content_seg_list
    contenttitle_seg = []
    if 'contenttitle' in xml_json:
        contenttitle = xml_json['contenttitle']
        contenttitle_seg = jieba.lcut(contenttitle)
    xml_json['contenttitle_seg'] = contenttitle_seg
    url = xml_json['cls_name']
    print '\t'.join([url,json.dumps(xml_json,ensure_ascii=False)])
